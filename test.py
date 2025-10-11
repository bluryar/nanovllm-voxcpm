from nanovllm.models.voxcpm.engine import VoxCPMEngine, VoxCPMConfig, Config
import os
import numpy as np
import soundfile as sf
import librosa

def main():
    path = os.path.expanduser("~/VoxCPM-0.5B")

    model_config = VoxCPMConfig.model_validate_json(open(os.path.join(path, "config.json")).read())
    engine_config = Config(
        model=path,
        model_config=model_config,
    )

    llm = VoxCPMEngine(engine_config)

    prompt_wav, _ = librosa.load("prompt.wav", sr=16000)
    print(prompt_wav.shape)
    llm.add_request(
        target_text="So what kind of music would you like to listen to today?",
        prompt_text="My name is kitty, and it's nice to meet you!",
        prompt_wav=prompt_wav,
        cfg_value=2,
    )

    output = llm.test_work()
    wav = np.concatenate(output[0].custom_payload.generated_waveforms, axis=0)

    sf.write("output.wav", wav, 16000)

if __name__ == "__main__":
    main()
