from nanovllm.engine.llm_engine import LLMEngineBase
from nanovllm.models.voxcpm.runner import VoxCPMRunner, RunnerTask, VoxCPMPayload
from nanovllm.config import Config
from nanovllm.models.voxcpm.config import VoxCPMConfig
from nanovllm.engine.sequence import Sequence
from dataclasses import dataclass
import numpy as np
from transformers import LlamaTokenizerFast
from nanovllm.models.voxcpm.utils import mask_multichar_chinese_tokens


@dataclass
class VoxCPMSeqPayload:
    # [(T, P, D)]
    feats : list[np.ndarray]

    text_tokens : list[int]
    feat_masks : list[bool]
    
    generated_waveforms : list[np.ndarray]

    temperature : float
    cfg_value : float

    decode_pad : np.ndarray | None = None
    max_generate_length : int | None = None
    

class VoxCPMEngine(LLMEngineBase):
    def __init__(self, config: Config[VoxCPMConfig]):
        self.n_decode_pad_frames = 3
        self.feat_dim = config.model_config.feat_dim
        self.patch_size = config.model_config.patch_size
        self.chunk_size = 640
        self.audio_start_token = 101
        self.block_size = config.kvcache_block_size

        self.tokenizer = mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(config.model))


        super().__init__(VoxCPMRunner, config, config.tensor_parallel_size)
    
    def preprocess_seq(self, seq: Sequence[VoxCPMSeqPayload], is_prefill: bool) -> RunnerTask[VoxCPMPayload]:
        if is_prefill:
            if len(seq.custom_payload.feats) > 1:
                feats = np.concatenate(seq.custom_payload.feats, axis=0)
                seq.custom_payload.feats = [feats]

            return RunnerTask(
                seq.block_table,
                len(seq),
                seq.num_cached_tokens,
                seq.block_size,
                VoxCPMPayload(
                    text_tokens=np.array(seq.custom_payload.text_tokens[seq.num_cached_tokens:], dtype=np.int64),
                    feats=seq.custom_payload.feats[-1][seq.num_cached_tokens:],
                    feat_masks=np.array(seq.custom_payload.feat_masks[seq.num_cached_tokens:], dtype=np.bool),
                    temperature=seq.custom_payload.temperature,
                    cfg_value=seq.custom_payload.cfg_value,
                    padding_decode=seq.custom_payload.decode_pad,
                )
            )
        else:
            return RunnerTask(
                seq.block_table,
                len(seq),
                len(seq) - 1,
                seq.block_size,
                VoxCPMPayload(
                    text_tokens=np.array(seq.custom_payload.text_tokens[-1:], dtype=np.int64),
                    feats=seq.custom_payload.feats[-1][-1:],
                    feat_masks=np.array(seq.custom_payload.feat_masks[-1:], dtype=np.bool),
                    temperature=seq.custom_payload.temperature,
                    cfg_value=seq.custom_payload.cfg_value,
                    padding_decode=seq.custom_payload.decode_pad,
                )
            )


    def postprocess_seq(self, seq: Sequence[VoxCPMSeqPayload], outputs: dict, is_prefill: bool):
        stop_flag = outputs["stop_flag"]
        latents = outputs["latents"]
        waveforms = outputs["waveforms"]

        seq.append_token(latents.tobytes())

        seq.custom_payload.feats.append(latents[None])
        seq.custom_payload.text_tokens.append(0)
        seq.custom_payload.feat_masks.append(True)

        seq.custom_payload.generated_waveforms.append(waveforms)

        latents = latents.reshape(-1, self.feat_dim)
        if seq.custom_payload.decode_pad is not None:
            seq.custom_payload.decode_pad = np.concatenate([seq.custom_payload.decode_pad, latents], axis=0)[-self.n_decode_pad_frames:]
        else:
            seq.custom_payload.decode_pad = latents[-self.n_decode_pad_frames:]

        if stop_flag == 1:
            seq.stoped = True
        elif seq.custom_payload.max_generate_length is not None and len(seq.custom_payload.generated_waveforms) >= seq.custom_payload.max_generate_length:
            seq.stoped = True

    def add_request(
            self,
            target_text : str,
            prompt_text : str = "",
            prompt_wav : np.ndarray = None,
            max_generate_length : int = 2000,
            temperature : float = 1.0,
            cfg_value : float = 1.0,
        ):
        text_tokens = self.tokenizer(prompt_text + target_text) + [self.audio_start_token]
        audio_feat = np.zeros((len(text_tokens), self.patch_size, self.feat_dim), dtype=np.float32)
        feat_masks = [False for _ in range(len(text_tokens))]
        hash_tokens = []
        for t in text_tokens:
            hash_tokens.append(t)

        decode_pad = None

        if prompt_wav is not None:
            if prompt_wav.ndim != 1:
                raise ValueError("prompt_wav must be 1D array")
            
            n_pad_to = self.patch_size * self.chunk_size
            if prompt_wav.shape[0] % n_pad_to != 0:
                remained = n_pad_to - prompt_wav.shape[0] % n_pad_to
                prompt_wav = np.pad(prompt_wav, (remained, 0), mode="constant", constant_values=0)

            wav_latents = self.model_runner.encode_latents(prompt_wav)
            decode_pad = wav_latents[-self.n_decode_pad_frames:]
            
            wav_latents = wav_latents.reshape(-1, self.patch_size, self.feat_dim)
            audio_feat = np.concatenate([audio_feat, wav_latents], axis=0)
            text_tokens.extend([0 for _ in range(wav_latents.shape[0])])
            feat_masks.extend([True for _ in range(wav_latents.shape[0])])

            for i in range(wav_latents.shape[0]):
                hash_tokens.append(wav_latents[i].tobytes())

        seq = Sequence(
            hash_tokens,
            self.block_size,
            VoxCPMSeqPayload(
                feats=[audio_feat],
                text_tokens=text_tokens,
                feat_masks=feat_masks,
                decode_pad=decode_pad,
                temperature=temperature,
                cfg_value=cfg_value,
                max_generate_length=max_generate_length,
                generated_waveforms=[],
            )
        )

        self.add_request_seq(seq)

    def test_work(self):
        outputs = {}
        while not self.is_finished():
            output = self.step()
        return output