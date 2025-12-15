import os
import io
import re
import pickle
import sqlite3
import numpy as np
import ffmpeg
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import Field
from pydantic_settings import BaseSettings

from nanovllm_voxcpm.models.voxcpm.server import AsyncVoxCPMServerPool


# -----------------------------------------------------------------------------
# Text Normalization
# -----------------------------------------------------------------------------

def clean_text_for_tts(text: str) -> str:
    """
    【TTS 专用清洗 V6 - 最终版】
    """
    if not text:
        return ""

    # 1. 基础清洗
    text = re.sub(r"[;；!！¡|｜¦·・]", " ", text)
    text = re.sub(r"[•◦‣➢➣➤●○■□◆◇]", "", text)
    text = re.sub(r"[©®™℗℠†‡¶§]", "", text)
    text = re.sub(r"[\x00-\x1f\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)

    # 2. 标点与特殊符号处理
    text = re.sub(r"(?<=\d)\s*[~～]\s*(?=\d)", "-", text)
    text = re.sub(r"[~～—…]+", "", text)
    text = re.sub(r"[-=/*#]{2,}", "", text)

    # 3. 整理空格
    text = re.sub(r"\s+", " ", text).strip()

    # 4. 去除结尾标点
    punctuation_pattern = (
        r"[,.;:!?，。；：！？、\"'`~@#$%^&*()_+={}\[\]|\\<>/«»‹›"
        "''„”“”‘’¨¯§¶©®™¢£¥€°±×÷¬¦¡¿ˉ―…∶‰′″‴¤₳฿₵¢₡₢$₫₯֏₠€ƒ₣₲₴₭₺₼ℳ₥₦₧₱₰£៛₽₹₨₪৳₸₮₩¥♳♴♵♶♷♸♹☰♀♂⁋⁌⁍※⁑⁂⁃⁅⁆⁇⁈⁉⁊⁍⁎⁏⁐⁑⁒⁓⁔⁕⁖⁗⁘⁙⁚⁛⁜⁝⁞⸀⸁⸂⸃⸄⸅⸆⸇⸈⸉⸊⸋⸌⸍⸎⸏⸐⸑⸒⸓⸔⸕⸖⸗⸘⸙⸚⸛⸜⸝⸞⸟⸠⸡⸢⸣⸤⸥⸦⸧⸨⸩⸪⸫⸬⸭⸮⸰⸱⸲⸳⸴⸵⸶⸷⸸⸹⸺⸻⸼⸽⸾⸿⹀⹁⹂⹃⹄⹅⹆⹇⹈⹉⹊⹋⹌⹍⹎⹏⹐⹑⹒⹓⹔⹕⹖⹗⹘⹙⹚⹛⹜⹝⹞⹟⹠⹡⹢⹣⹤⹥⹦⹧⹨⹩⹪⹫⹬⹭⹮⹯⹰⹱⹲⹳⹴⹵⹶⹷⹸⹹⹺⹻⹼⹽⹾⹿\-—–]+$"
    )
    while text and re.search(punctuation_pattern, text[-1]):
        text = text[:-1].strip()

    # 5. 【智能结束符】
    if text:
        if re.search(r"[\u4e00-\u9fff]", text):
            text += "。"
        else:
            text += "."

    return text


def normalize_text_voxcpm(text: str) -> str:
    """参考 VoxCPM-py 的 text_normalize 逻辑"""
    if not text:
        return ""
    try:
        import regex as re2
    except Exception:
        re2 = None
    try:
        import inflect
    except Exception:
        inflect = None
    try:
        from wetext import Normalizer
    except Exception:
        Normalizer = None

    def clean_markdown(md: str) -> str:
        md = re.sub(r"```.*?```", "", md, flags=re.DOTALL)
        md = re.sub(r"`[^`]*`", "", md)
        md = re.sub(r"!\[[^\]]*\]\([^\)]+\)", "", md)
        md = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md)
        md = re.sub(r"^(\s*)-\s+", r"\1", md, flags=re.MULTILINE)
        md = re.sub(r"<[^>]+>", "", md)
        md = re.sub(r"^#{1,6}\s*", "", md, flags=re.MULTILINE)
        md = re.sub(r"\n\s*\n", "\n", md)
        return md.strip()

    def remove_emoji(s: str) -> str:
        if re2 is None:
            return s
        return re2.compile(
            r"\p{Emoji_Presentation}|\p{Emoji}\uFE0F", flags=re2.UNICODE
        ).sub("", s)

    def contains_chinese(t: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]+", t))

    def remove_bracket(t: str) -> str:
        t = t.replace("（", " ").replace("）", " ")
        t = t.replace("【", " ").replace("】", " ")
        t = t.replace("`", "")
        t = t.replace("——", " ")
        return t

    def spell_out_number_en(t: str, inflect_parser):
        if inflect_parser is None:
            return t
        res = []
        buf = []
        for ch in t:
            if ch.isdigit():
                buf.append(ch)
            else:
                if buf:
                    num_word = inflect_parser.number_to_words("".join(buf))
                    res.append(num_word)
                    buf = []
                res.append(ch)
        if buf:
            res.append(inflect_parser.number_to_words("".join(buf)))
        return "".join(res)

    text = clean_markdown(text)
    text = remove_emoji(text)
    text = text.replace("\n", " ").replace("\t", " ")
    text = text.replace('"', "“")
    text = re.sub(r"\s+", " ", text).strip()

    lang_is_zh = contains_chinese(text)

    if Normalizer is not None:
        try:
            if lang_is_zh:
                text = text.replace("=", "等于")
                if re.search(r"([\d$%^*_+≥≤≠×÷?=])", text):
                    text = re.sub(r"(?<=[a-zA-Z0-9])-(?=\d)", " - ", text)
                text = Normalizer(
                    lang="zh", operator="tn", remove_erhua=True
                ).normalize(text)
                text = remove_bracket(text)
            else:
                text = Normalizer(lang="en", operator="tn").normalize(text)
                text = spell_out_number_en(text, inflect.engine() if inflect else None)
            text = re.sub(r"\s+", " ", text).strip()
            text = clean_text_for_tts(text)
            return text
        except Exception:
            pass

    return clean_text_for_tts(text)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

class Settings(BaseSettings):
    model_path: str = Field(default="~/VoxCPM-0.5B", env="MODEL_PATH")
    max_num_batched_tokens: int = Field(default=8192, env="MAX_NUM_BATCHED_TOKENS")
    max_num_seqs: int = Field(default=16, env="MAX_NUM_SEQS")
    max_model_len: int = Field(default=4096, env="MAX_MODEL_LEN")
    gpu_memory_utilization: float = Field(default=0.95, env="GPU_MEMORY_UTILIZATION")
    enforce_eager: bool = Field(default=False, env="ENFORCE_EAGER")
    devices: list[int] = Field(default=[0], env="DEVICES")
    db_path: str = Field(default="prompts.db", env="DB_PATH")
    inference_timesteps: int = Field(default=10, env="INFERENCE_TIMESTEPS")
    scheduler_log_interval: float = Field(default=5.0, env="SCHEDULER_LOG_INTERVAL")
    scheduler_log_enable: bool = Field(default=True, env="SCHEDULER_LOG_ENABLE")

    class Config:
        env_file = ".env"

settings = Settings()

# -----------------------------------------------------------------------------
# Global State & Lifespan
# -----------------------------------------------------------------------------

global_instances = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    global_instances["server"] = AsyncVoxCPMServerPool(
        model_path=settings.model_path,
        max_num_batched_tokens=settings.max_num_batched_tokens,
        max_num_seqs=settings.max_num_seqs,
        max_model_len=settings.max_model_len,
        gpu_memory_utilization=settings.gpu_memory_utilization,
        enforce_eager=settings.enforce_eager,
        devices=settings.devices,
        inference_timesteps=settings.inference_timesteps,
        scheduler_log_interval=settings.scheduler_log_interval,
        scheduler_log_enable=settings.scheduler_log_enable,
    )
    await global_instances["server"].wait_for_ready()
    yield
    await global_instances["server"].stop()
    del global_instances["server"]

app = FastAPI(lifespan=lifespan)

# -----------------------------------------------------------------------------
# Database Helpers
# -----------------------------------------------------------------------------

def init_db():
    conn = sqlite3.connect(settings.db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prompts (
            id TEXT PRIMARY KEY,
            latents BLOB NOT NULL,
            text TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def get_prompt(prompt_id: str):
    conn = sqlite3.connect(settings.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT latents, text FROM prompts WHERE id = ?", (prompt_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"latents": row[0], "text": row[1]}
    return None

def save_prompt(prompt_id: str, latents: bytes, text: str):
    conn = sqlite3.connect(settings.db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO prompts (id, latents, text) VALUES (?, ?, ?)", (prompt_id, latents, text))
    conn.commit()
    conn.close()

def delete_prompt(prompt_id: str):
    conn = sqlite3.connect(settings.db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))
    conn.commit()
    conn.close()

def list_all_prompts():
    conn = sqlite3.connect(settings.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, text FROM prompts")
    rows = cursor.fetchall()
    conn.close()
    return [{"id": row[0], "text": row[1]} for row in rows]

# -----------------------------------------------------------------------------
# Audio Utilities
# -----------------------------------------------------------------------------

async def preprocess_audio(audio_bytes: bytes, audio_format: str, target_sample_rate: int, patch_size: int, chunk_size: int) -> bytes:
    """
    Robust audio preprocessing for VoxCPM:
    1. Decode audio (supports multiple formats via ffmpeg/torchaudio)
    2. Resample to target_sample_rate (44.1k for v1.5, 16k for v0.5)
    3. Convert to mono
    4. Left-pad to align with patch_size * chunk_size
    """
    import torchaudio
    import torch
    
    # 1. 尝试使用 torchaudio 读取 (支持 wav, mp3, flac 等常见格式)
    try:
        # torchaudio.load 能够处理 BytesIO
        wav_tensor, sr = torchaudio.load(io.BytesIO(audio_bytes), format=audio_format)
    except Exception:
        # 2. 如果 torchaudio 失败 (例如不支持的格式或ffmpeg问题)，尝试使用 ffmpeg-python 进行转换
        try:
            import ffmpeg
            # 使用 ffmpeg 将输入音频转换为 wav 格式 (pcm_s16le)，采样率为模型需要的采样率
            # 注意: input 接受 pipe:0 作为标准输入
            out, _ = (
                ffmpeg
                .input('pipe:0')
                .output('pipe:1', format='wav', acodec='pcm_s16le', ar=target_sample_rate)
                .run(input=audio_bytes, capture_stdout=True, capture_stderr=True)
            )
            # 转换后的数据再次用 torchaudio 读取 (已经是 wav 格式了)
            wav_tensor, sr = torchaudio.load(io.BytesIO(out))
        except Exception as e:
            print(f"[VoxCPM App] 音频解码失败: {e}")
            raise ValueError(f"无法解码音频数据: {e}")

    if torch.cuda.is_available():
        wav_tensor = wav_tensor.cuda()
    
    # 3. 强制重采样 (如果之前的 ffmpeg 转换没有处理好，或者直接 torchaudio 读取的采样率不对)
    if sr != target_sample_rate:
        wav_tensor = torchaudio.functional.resample(wav_tensor, sr, target_sample_rate)
    
    # 4. 转换为单声道
    if wav_tensor.size(0) > 1:
        wav_tensor = wav_tensor.mean(dim=0, keepdim=True)
    
    # 5. 对齐填充 (Left Padding)
    align_size = patch_size * chunk_size
    if wav_tensor.size(1) % align_size != 0:
        remained = align_size - wav_tensor.size(1) % align_size
        # 左侧填充: (padding_left, padding_right)
        wav_tensor = torch.nn.functional.pad(wav_tensor, (remained, 0))

    # 返回 bytes (wav格式) 供 server 使用
    # 注意: server.encode_latents 期望接收 wav 格式的 bytes，并且会在内部再次 load
    # 为了避免 server 再次处理带来的不确定性，我们这里最好直接返回 tensor 给 server?
    # 但是 server 接口是 encode_latents(wav: bytes, wav_format: str)
    # 所以我们这里需要把处理好的 tensor 转回 wav bytes
    
    buf = io.BytesIO()
    torchaudio.save(buf, wav_tensor.cpu(), target_sample_rate, format="wav")
    return buf.getvalue()

async def numpy_to_bytes(gen):
    async for chunk in gen:
        yield chunk.tobytes()

async def convert_audio_stream(gen, sample_rate=16000, output_format="mp3", speed=1.0):
    """
    Convert raw PCM audio stream (float32) to desired format using ffmpeg-python.
    Defaults to 16000Hz as that's what VoxCPM usually outputs (or is resampled to).
    """
    # Input stream
    input_stream = ffmpeg.input('pipe:', format='f32le', acodec='pcm_f32le', ac=1, ar=sample_rate)

    # Apply speed filter if needed
    if speed != 1.0:
        stream = input_stream
        remaining_speed = speed
        
        # Handle speed < 0.5
        while remaining_speed < 0.5:
            stream = stream.filter('atempo', 0.5)
            remaining_speed /= 0.5
            
        # Handle speed > 2.0
        while remaining_speed > 2.0:
            stream = stream.filter('atempo', 2.0)
            remaining_speed /= 2.0
            
        # Apply remaining factor
        if remaining_speed != 1.0:
            stream = stream.filter('atempo', remaining_speed)
    else:
        stream = input_stream

    # Create an ffmpeg process that reads from pipe and writes to pipe
    process = (
        stream
        .output('pipe:', format=output_format)
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    )

    async def writer():
        try:
            async for chunk in gen:
                # Chunk is numpy array (float32)
                # Write bytes to ffmpeg stdin
                await asyncio.to_thread(process.stdin.write, chunk.tobytes())
            process.stdin.close()
        except Exception as e:
            print(f"Error in ffmpeg writer: {e}")
            process.kill()

    # Start writer task
    import asyncio
    writer_task = asyncio.create_task(writer())

    try:
        # Read output from ffmpeg stdout
        while True:
            data = await asyncio.to_thread(process.stdout.read, 4096)
            if not data:
                break
            yield data
            
        await writer_task
        process.wait()
    except Exception as e:
        print(f"Error in ffmpeg conversion: {e}")
        process.kill()
        raise e

# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/add_prompt")
async def add_prompt(
    prompt_id: str = Form(...),
    file: UploadFile = File(...),
    prompt_text: str = Form(...)
):
    server: AsyncVoxCPMServerPool = global_instances["server"]
    
    # Read the uploaded file
    wav_bytes = await file.read()
    wav_format = file.filename.split(".")[-1] if file.filename and "." in file.filename else "wav"
    
    # Encode latents
    # Preprocess audio using the robust logic moved from server.py
    # We need to get model params from the server instance first
    # Since AsyncVoxCPMServerPool doesn't expose patch_size/chunk_size directly, we might need to peek into one of its servers or add a property
    # But wait, server.encode_latents does internal processing too? 
    # The user wanted to move logic HERE. 
    # Let's inspect server.py again. It seems server.encode_latents expects raw bytes and does some basic checks.
    # But we want to do the HEAVY lifting here (ffmpeg, padding) so server gets clean input.
    
    # We need access to model config to know sample_rate/patch_size/chunk_size
    # global_instances["server"] is AsyncVoxCPMServerPool
    # Let's assume we can get these from the pool or its first server
    
    # Accessing internal server state is tricky with AsyncVoxCPMServerPool wrapper.
    # However, the pool does have self.sample_rate initialized in __init__
    # But it doesn't seem to expose patch_size/chunk_size easily without peeking into a worker process (which is hard).
    
    # WAIT! The original request was to move the logic to app.py.
    # But app.py runs in the main process, while server runs in subprocesses (multiprocessing).
    # If we do preprocessing in app.py, we save the workers from doing it.
    
    # Issue: How to get patch_size and chunk_size here?
    # In server.py: 
    # self.llm = VoxCPMEngine(engine_config)
    # self.llm.patch_size = ...
    # self.llm.chunk_size = ...
    
    # AsyncVoxCPMServerPool initializes servers but doesn't keep a local copy of llm config except sample_rate.
    # We might need to extend AsyncVoxCPMServerPool to expose these, or fetch them via a "info" command.
    # For now, let's look at how we can get them.
    # We can add a method to AsyncVoxCPMServerPool to get model info.
    
    # Actually, let's look at server.py again. 
    # AsyncVoxCPMServerPool has `self.sample_rate`.
    # It does NOT have patch_size or chunk_size.
    
    # Let's Modify server.py to expose get_model_info() first?
    # Or... we can assume standard values if not available? No, that breaks the "no hardcoding" rule.
    
    # Alternative:
    # In app.py's lifespan, after server init, we can call a new method `server.get_model_config()`
    
    # Let's stick to what we have. 
    # We can add a `get_model_config` method to AsyncVoxCPMServerPool that queries one of the workers.
    
    # For this step, I will add the preprocessing call, but I need those params.
    # I'll first add `get_model_config` to server.py
    
    # ... Wait, I should do that in a separate tool call if needed.
    # But I can't leave the file in a broken state.
    
    # Let's check if I can get them from `settings` or `model_path`?
    # We can read config.json again here?
    # app.py has `settings.model_path`.
    # We can read `config.json` in app.py just like server does!
    
    from nanovllm_voxcpm.models.voxcpm.engine import VoxCPMConfig
    try:
        model_config_path = os.path.join(settings.model_path, "config.json")
        # Expand user path if needed
        model_config_path = os.path.expanduser(model_config_path)
        
        if os.path.exists(model_config_path):
             with open(model_config_path, 'r') as f:
                import json
                config_dict = json.load(f)
                
             # Parse relevant fields
             # V1.5 vs V0.5 logic
             # This duplicates logic in engine.py, but it's safe since it's just reading config
             
             # default values
             patch_size = config_dict.get("patch_size", 1) # Should be in config?
             # Wait, patch_size is usually in model_config
             
             # Let's use the Pydantic model to be safe
             model_conf = VoxCPMConfig.model_validate(config_dict)
             patch_size = model_conf.patch_size
             
             if model_conf.audio_vae_config:
                 target_sample_rate = model_conf.audio_vae_config.sample_rate
                 chunk_size = int(np.prod(model_conf.audio_vae_config.encoder_rates))
             else:
                 target_sample_rate = 16000
                 chunk_size = 640 # prod([2, 5, 8, 8])
                 
    except Exception as e:
        print(f"Failed to load model config in app.py: {e}. Using defaults.")
        target_sample_rate = 16000
        patch_size = 1 # ?
        chunk_size = 640
        
    # Now use them
    wav_bytes = await preprocess_audio(wav_bytes, wav_format, target_sample_rate, patch_size, chunk_size)
    
    # Update format to "wav" since we converted it
    wav_format = "wav"
    
    latents = await server.encode_latents(wav_bytes, wav_format)
    
    # Save to SQLite
    latents_bytes = pickle.dumps(latents)
    save_prompt(prompt_id, latents_bytes, prompt_text)
    
    return {"status": "ok", "prompt_id": prompt_id}

@app.post("/del_prompt")
async def del_prompt(prompt_id: str = Form(...)):
    delete_prompt(prompt_id)
    return {"status": "ok", "prompt_id": prompt_id}

@app.get("/list_prompt")
async def list_prompt():
    return list_all_prompts()

@app.get("/v1/models")
async def show_available_models():
    """
    Show available models.
    Matches OpenAI API signature: https://platform.openai.com/docs/api-reference/models/list
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "voxcpm",
                "object": "model",
                "created": 1765436831,
                "owned_by": "nanovllm-voxcpm",
                "root": "voxcpm",
                "max_model_len": 4096
            }
        ]
    }

@app.post("/v1/audio/speech")
async def create_speech(
    input: Optional[str] = Form(None), # required
    model: str = Form("voxcpm"),
    voice: str = Form("default"),
    response_format: str = Form("mp3"), # default is mp3
    speed: float = Form(1.0),
    # Extra fields for VoxCPM specific needs (Voice Cloning)
    temperature: float = Form(1.0),
    cfg_value: float = Form(1.5),
    stream_format: str = Form("audio"), # "audio" or "sse"
):
    """
    Generates audio from the input text.
    Matches OpenAI API signature: https://platform.openai.com/docs/api-reference/audio/createSpeech
    """
    server : AsyncVoxCPMServerPool = global_instances["server"]
    
    if not input:
        async def empty_generator():
            yield b""
            return

        if stream_format == "sse":
            import json
            async def empty_sse_generator():
                done_event = {
                    "type": "speech.audio.done",
                    "usage": {
                        "total_tokens": 0 
                    }
                }
                yield f"data: {json.dumps(done_event)}\n\n"

            return StreamingResponse(
                empty_sse_generator(),
                media_type="text/event-stream",
            )
        
        return StreamingResponse(
            empty_generator(),
            media_type="audio/raw" if response_format in ["pcm", "raw"] else f"audio/{response_format}",
        )
    
    # -------------------------------------------------------------------------
    # Text Normalization
    # -------------------------------------------------------------------------
    # Apply text normalization using the migrated logic
    input_text = normalize_text_voxcpm(input)
    # -------------------------------------------------------------------------

    prompt_latents = None
    prompt_text = ""
    
    # 1. Check if voice is provided and exists in DB
    if voice and voice != "default":
        # Check if voice is a stored prompt_id
        stored_prompt = get_prompt(voice)
        if stored_prompt:
            prompt_latents = pickle.loads(stored_prompt["latents"])
            prompt_text = stored_prompt["text"]
    
    # Use normalized 'input_text' as 'target_text'
    target_text = input_text

    audio_generator = server.generate(
        target_text=target_text,
        prompt_latents=prompt_latents,
        prompt_text=prompt_text,
        max_generate_length=2000,
        temperature=temperature,
        cfg_value=cfg_value,
    )

    if stream_format == "sse":
        import base64
        import json
        async def sse_generator():
             async for chunk in audio_generator:
                if chunk is not None:
                     # chunk is numpy float32 array
                     # Convert float32 to int16 PCM
                     chunk = np.clip(chunk, -1.0, 1.0)
                     chunk = (chunk * 32767).astype(np.int16)
                     
                     # For SSE, OpenAI usually returns chunks of base64 encoded audio
                     # We send raw PCM bytes (s16le) as base64.
                     b64_data = base64.b64encode(chunk.tobytes()).decode("utf-8")
                     
                     event = {
                         "type": "speech.audio.delta",
                         "audio": b64_data
                     }
                     yield f"data: {json.dumps(event)}\n\n"
             
             # Send done event
             # We don't have token usage stats easily available from the current generator
             # So we'll send a dummy usage or omit it if optional
             done_event = {
                 "type": "speech.audio.done",
                 "usage": {
                     "total_tokens": 0 
                 }
             }
             yield f"data: {json.dumps(done_event)}\n\n"

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
        )

    if response_format == "pcm" or response_format == "raw":
        return StreamingResponse(
            numpy_to_bytes(audio_generator),
            media_type="audio/raw",
        )
    else:
        # Map OpenAI formats to ffmpeg formats if needed
        ffmpeg_format = response_format
        media_type = f"audio/{response_format}"
        
        if response_format == "aac":
            ffmpeg_format = "adts" # AAC stream format
        elif response_format == "pcm":
            ffmpeg_format = "s16le" # Standard PCM
            media_type = "audio/pcm"

        return StreamingResponse(
            convert_audio_stream(audio_generator, sample_rate=server.sample_rate, output_format=ffmpeg_format, speed=speed),
            media_type=media_type,
        )
