from nanovllm_voxcpm.models.voxcpm.engine import VoxCPMEngine, VoxCPMConfig, Config
import os
import torch.multiprocessing as mp
from queue import Empty
import traceback
import uuid
import torchaudio
import io
import torch
import asyncio
from typing import List
import numpy as np
import random
import threading
import time

def gen_uuid() -> str:
    return uuid.uuid4().hex

class VoxCPMServerImpl:
    def __init__(self,
        model_path : str,
        inference_timesteps : int = 10,
        max_num_batched_tokens : int = 16384,
        max_num_seqs : int = 512,
        max_model_len : int = 4096,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        devices : List[int] = [],
    ):
        model_config = VoxCPMConfig.model_validate_json(
            open(os.path.join(model_path, "config.json")).read()
        )

        model_config.inference_timesteps = inference_timesteps

        engine_config = Config(
            model=model_path,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            model_config=model_config,
            devices=devices,
        )

        self.llm = VoxCPMEngine(engine_config)
        self.sample_rate = self.llm.model_runner.vae.sample_rate

    def health(self):
        return {
            "status": "ok",
        }
    
    def encode_latents(self, wav : bytes, wav_format : str):
        wav, sr = torchaudio.load(io.BytesIO(wav), format=wav_format)
        wav = wav.cuda()
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        latents = self.llm.encode_latents(wav)
        assert latents.shape[0] % self.llm.patch_size == 0
        
        return latents.tobytes()

    def add_request(self,
        seq_id : str,
        target_text : str,
        prompt_latents : bytes | None = None,
        prompt_text : str = "",
        max_generate_length : int = 2000,
        temperature : float = 1.0,
        cfg_value : float = 1.0
    ):
        if prompt_latents is not None:
            if len(prompt_text) == 0:
                raise ValueError("Prompt text is required when prompt latents are provided")
            
            prompt_latents = np.frombuffer(prompt_latents, dtype=np.float32).reshape(-1, self.llm.feat_dim)
        else:
            prompt_latents = None
            if len(prompt_text) > 0:
                raise ValueError("Prompt text is not allowed when prompt latents are not provided")
        
        self.llm.add_request(
            seq_id=seq_id,
            target_text=target_text,
            prompt_text=prompt_text,
            prompt_latents=prompt_latents,
            max_generate_length=max_generate_length,
            temperature=temperature,
            cfg_value=cfg_value,
        )

    def cancel(self, seq_id : str):
        self.llm.cancel_sequence(seq_id)
    
    def step(self):
        return self.llm.step()
    
    def is_finished(self):
        return self.llm.is_finished()


def main_loop(
    queue_in : mp.Queue,
    queue_out : mp.Queue,
    args, kwargs,
    scheduler_log_interval : float = 5.0,
    scheduler_log_enable : bool = True
):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    srv = VoxCPMServerImpl(*args, **kwargs)

    states = {
        "is_stoped": False,
    }
    def method_call(cmd):
        try:
            opid = cmd["id"]
            method_name = cmd["type"]
            args = cmd["args"]
            kwargs = cmd["kwargs"]

            if method_name == "stop":
                states["is_stoped"] = True
                try:
                    printer_stop.set()
                except Exception:
                    pass
                return {
                    "type": "response",
                    "id": opid,
                    "data": None,
                }

            ret = getattr(srv, method_name)(*args, **kwargs)
            return {
                "type": "response",
                "id": opid,
                "data": ret,
            }
        except Exception:
            traceback_str = traceback.format_exc()
            return {
                "type": "error",
                "id": opid,
                "error": traceback_str,
            }

    printer_stop = threading.Event()
    def printer_loop():
        interval = scheduler_log_interval
        enable_log = scheduler_log_enable

        if not enable_log:
            return

        while not printer_stop.is_set():
            try:
                stats = srv.llm.scheduler.stats()
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f"[{timestamp}] [ServerStats] waiting={stats['waiting']} running={stats['running']}")
            except Exception:
                pass
            time.sleep(interval)

    printer_thread = threading.Thread(target=printer_loop, daemon=True)
    printer_thread.start()

    while not states["is_stoped"]:
        # while llm server is empty
        cmd = queue_in.get()
        queue_out.put(method_call(cmd))

        while not srv.is_finished() and not states["is_stoped"]:
            # while llm server is not empty
            while not states["is_stoped"]:
                # get cmd nowait, and handle it first
                try:
                    cmd = queue_in.get_nowait()
                    queue_out.put(method_call(cmd))
                except Empty:
                    break
            
            if states["is_stoped"]:
                break
            
            output = srv.step()

            # update output
            for seq in output:
                latest_waveform = seq.custom_payload.generated_waveforms[-1]
                queue_out.put({
                    "type": "stream",
                    "id": seq.seq_id,
                    "data": latest_waveform,
                })
                if seq.is_finished:
                    queue_out.put({
                        "type": "stream",
                        "id": seq.seq_id,
                        "data": None,
                    })

    try:
        printer_stop.set()
        printer_thread.join(timeout=2)
    except Exception:
        pass


class AsyncVoxCPMServer:
    def __init__(self,
        model_path : str,
        inference_timesteps : int = 10,
        max_num_batched_tokens : int = 16384,
        max_num_seqs : int = 512,
        max_model_len : int = 4096,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        devices : List[int] = [],
        scheduler_log_interval : float = 5.0,
        scheduler_log_enable : bool = True,
        **kwargs,
    ):
        if len(kwargs) > 0:
            raise ValueError(f"Unknown kwargs: {kwargs}")

        ctx = mp.get_context("spawn")
        self.queue_in = ctx.Queue()
        self.queue_out = ctx.Queue()
        self.process = ctx.Process(
            target=main_loop, 
            args=(self.queue_in, self.queue_out, (model_path, inference_timesteps, max_num_batched_tokens, max_num_seqs, max_model_len, gpu_memory_utilization, enforce_eager, devices), {}, scheduler_log_interval, scheduler_log_enable),
            daemon=True,
        )
        self.process.start()

        self.recv_task = asyncio.create_task(self.recv_queue())
        self.op_table = {}
        self.stream_table : dict[str, asyncio.Queue] = {}
    
    async def recv_queue(self):
        while True:
            try:
                res = await asyncio.to_thread(self.queue_out.get, timeout=1)
            except Empty:
                continue

            if res["type"] == "stream":
                if res["id"] in self.stream_table:
                    stream_data = res["data"]
                    await self.stream_table[res["id"]].put(stream_data)
                else:
                    print(f"Unknown stream_id: {res['id']}")
            elif res["id"] in self.op_table:
                if res["type"] == "response":
                    self.op_table[res["id"]].set_result(res["data"] if "data" in res else None)
                    del self.op_table[res["id"]]
                else:
                    self.op_table[res["id"]].set_exception(RuntimeError(res["error"]))
                    del self.op_table[res["id"]]
            else:
                print(f"Unknown op_id: {res['id']}")
    
    async def submit(self, cmd : str, *args, **kwargs):
        op_id = str(uuid.uuid4())

        loop = asyncio.get_running_loop()
        fut = loop.create_future()

        self.op_table[op_id] = fut

        await asyncio.to_thread(self.queue_in.put, {
            "id": op_id,
            "type": cmd,
            "args": args,
            "kwargs": kwargs,
        })
        return await fut
    
    async def health(self):
        return await self.submit("health")
    
    async def wait_for_ready(self):
        return await self.health()
    
    async def encode_latents(self, wav : bytes, wav_format : str):
        return await self.submit("encode_latents", wav, wav_format)
    
    async def stop(self):
        await self.submit("stop")
        self.recv_task.cancel()
        await asyncio.to_thread(self.process.join)
    
    async def generate(
        self,
        target_text : str,
        prompt_latents : bytes | None = None,
        prompt_text : str = "",
        max_generate_length : int = 2000,
        temperature : float = 1.0,
        cfg_value : float = 2.0
    ):
        seq_id = gen_uuid()
        self.stream_table[seq_id] = asyncio.Queue()

        is_normal_exit = False
        try:
            await self.submit("add_request", seq_id, target_text, prompt_latents, prompt_text, max_generate_length, temperature, cfg_value)

            while True:
                data = await self.stream_table[seq_id].get()
                if data is None:
                    is_normal_exit = True
                    break
                yield data
        finally:
            if not is_normal_exit:
                await self.submit("cancel", seq_id)
            del self.stream_table[seq_id]


class AsyncVoxCPMServerPool:
    def __init__(self,
        model_path : str,
        inference_timesteps : int = 10,
        max_num_batched_tokens : int = 16384,
        max_num_seqs : int = 512,
        max_model_len : int = 4096,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        devices : List[int] = [],
        scheduler_log_interval : float = 5.0,
        scheduler_log_enable : bool = True,
        **kwargs,
    ):
        if len(kwargs) > 0:
            raise ValueError(f"Unknown kwargs: {kwargs}")

        self.servers = [
            AsyncVoxCPMServer(
                model_path=model_path,
                inference_timesteps=inference_timesteps,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=enforce_eager,
                devices=[device_idx],
                scheduler_log_interval=scheduler_log_interval,
                scheduler_log_enable=scheduler_log_enable,
            )
            for device_idx in devices
        ]
        
        self.servers_load = np.zeros(len(self.servers), dtype=np.int32)
        
        model_config = VoxCPMConfig.model_validate_json(
            open(os.path.join(model_path, "config.json")).read()
        )
        # 如果配置里没有 sample_rate，则默认 16000，避免程序崩溃
        try:
            self.sample_rate = model_config.audio_vae_config.sample_rate
        except AttributeError:
            self.sample_rate = 16000
        self._prompt_pool = {}

    async def wait_for_ready(self):
        await asyncio.gather(*[server.wait_for_ready() for server in self.servers])
    
    async def stop(self):
        await asyncio.gather(*[server.stop() for server in self.servers])
    
    async def encode_latents(self, wav : bytes, wav_format : str):
        # send to one
        min_load_server_idx = np.argmin(self.servers_load)
        return await self.servers[min_load_server_idx].encode_latents(wav, wav_format)
    
    async def generate(
        self,
        target_text : str,
        prompt_latents : bytes | None = None,
        prompt_text : str = "",
        max_generate_length : int = 2000,
        temperature : float = 1.0,
        cfg_value : float = 2.0
    ):
        min_load_server_idx = np.argmin(self.servers_load)
        self.servers_load[min_load_server_idx] += 1

        server = self.servers[min_load_server_idx]

        try:
            async for data in server.generate(target_text, prompt_latents, prompt_text, max_generate_length, temperature, cfg_value):
                yield data
        finally:
            self.servers_load[min_load_server_idx] -= 1

class SyncVoxCPMServerPool:
    def __init__(self, 
            model_path : str,
            inference_timesteps : int = 10,
            max_num_batched_tokens : int = 16384,
            max_num_seqs : int = 512,
            max_model_len : int = 4096,
            gpu_memory_utilization: float = 0.9,
            enforce_eager: bool = False,
            devices : List[int] = [],
            **kwargs,
        ):
        async def init_async_server_pool():
            return AsyncVoxCPMServerPool(
                model_path=model_path,
                inference_timesteps=inference_timesteps,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=enforce_eager,
                devices=devices,
                **kwargs,
            )

        self.loop = asyncio.new_event_loop()
        self.server_pool = self.loop.run_until_complete(init_async_server_pool())
        self.loop.run_until_complete(self.server_pool.wait_for_ready())
    
    def stop(self):
        self.loop.run_until_complete(self.server_pool.stop())
        self.loop.close()
        self.loop = None

    def encode_latents(self, wav : bytes, wav_format : str):
        return self.loop.run_until_complete(self.server_pool.encode_latents(wav, wav_format))
    
    def generate(self, target_text : str, prompt_latents : bytes | None = None, prompt_text : str = "", max_generate_length : int = 2000, temperature : float = 1.0, cfg_value : float = 2.0):
        async_gen = self.server_pool.generate(target_text, prompt_latents, prompt_text, max_generate_length, temperature, cfg_value)
        try:
            while True:
                item = self.loop.run_until_complete(async_gen.__anext__())
                yield item
        except StopAsyncIteration:
            return

    
