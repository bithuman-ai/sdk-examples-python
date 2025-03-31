from typing import Iterable

import numpy as np
import soundfile as sf

from bithuman.api import AudioChunk
from bithuman.audio.utils import (
    AudioStreamBatcher,
    float32_to_int16,
    load_audio,
)

audio_np, sr = load_audio("python_module/examples/sample.wav", 16000)
audio_np = float32_to_int16(audio_np)

batcher = AudioStreamBatcher()


def audio_chunk_generator(
    audio_np: np.ndarray, chunk_size: int
) -> Iterable[bytes | None]:
    for i in range(0, len(audio_np), chunk_size):
        yield audio_np[i : i + chunk_size].tobytes()
    yield None


print(f"input_audio.shape: {audio_np.shape}, duration: {len(audio_np) / sr} seconds")
print(f"input_frames: {len(audio_np) / (sr // 25)}")
new_audio = []
for audio_bytes in audio_chunk_generator(audio_np, sr // 100):
    padded_chunks = batcher.push(
        AudioChunk.from_bytes(audio_bytes, sr, last_chunk=False)
        if audio_bytes
        else None
    )
    padded_chunks = list(padded_chunks)
    for padded_chunk in padded_chunks:
        # print(f"padded_chunk.array.shape: {padded_chunk.array.shape}")
        audio_array = batcher.unpad(padded_chunk.array)
        new_audio.append(audio_array)

new_audio = np.concatenate(new_audio)
print(f"output_audio.shape: {new_audio.shape}, duration: {len(new_audio) / sr} seconds")
print(f"output_frames: {len(new_audio) / (16000 // 25)}")

sf.write("output.wav", new_audio, 16000)
