# %%
import sys
import time
from typing import Iterable

import numpy as np
from loguru import logger
from tqdm import tqdm

from bithuman import Bithuman
from bithuman.api import AudioChunk, VideoControl, VideoFrame
from bithuman.audio import (
    float32_to_int16,
    int16_to_float32,
    load_audio,
    write_video_with_audio,
)
from bithuman.config import load_settings

logger.remove()
logger.add(sys.stderr, level="INFO")

# %%
settings = load_settings()
settings.OUTPUT_WIDTH = 1280
settings.COMPRESS_METHOD = "JPEG"


runtime = Bithuman()
runtime.validate_token(
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJiaXRodW1hbiIsImlhdCI6MTc0MDg0MjY1NiwiZXhwIjoxNzQwODQ2MjU2fQ.OtdEwUkr5L1DVZpHqKo1KDWJ9dVlXp4fs_S4bNwsW4E"
)
runtime.set_model(
    model_path="/Volumes/razer-data/cpu-asserts/Hysses0126/Hysses20250126.bhmodel"
)


# %%

audio_fp32, sr = load_audio(
    "/home/longc/data/code/bithuman/python_module/examples/sample.wav",
    target_sr=16000,
)

# %%
# Generate video
TEST_VIDEO_GEN = False
if TEST_VIDEO_GEN:
    controls = [VideoControl.from_audio(audio_fp32, sr)]
    all_frames: list[VideoFrame] = []
    for audio_data in controls:
        gen_frames = runtime.process(audio_data)
        all_frames.extend(list(tqdm(gen_frames)))

    audio_np = np.concatenate([audio_data.audio.array for audio_data in controls])
    images = [frame.rgb_image for frame in all_frames]
    write_video_with_audio(
        "test_single_process.mp4",
        images,
        int16_to_float32(audio_np),
        sr,
        fps=25,
    )
# %%
# Simulate streaming

# repeat audio to simulate streaming
audio_np = float32_to_int16(audio_fp32)
audio_np = np.repeat(audio_np, 10, axis=0)
duration = len(audio_np) / sr
print(f"Duration: {duration} seconds")


def audio_chunk_generator(audio_np: np.ndarray, chunk_size: int) -> Iterable[bytes]:
    for i in range(0, len(audio_np), chunk_size):
        yield audio_np[i : i + chunk_size].tobytes()
    yield None


tic = time.time()
all_gen_frames: list[VideoFrame] = []
total_frames = 0
with tqdm() as pbar:
    for audio_data in audio_chunk_generator(audio_np, sr // 100):
        audio_chunk = (
            AudioChunk.from_bytes(audio_data, sr, last_chunk=False)
            if audio_data
            else None
        )
        for frame in runtime.process(VideoControl(audio_chunk)):
            # all_gen_frames.append(frame)
            total_frames += 1
            pbar.update(1)
toc = time.time()

print(
    f"Time taken: {toc - tic} seconds, frames: {total_frames}, FPS: {total_frames / (toc - tic)}"
)

# %%

if all_gen_frames:
    audio_output = np.concatenate(
        [frame.audio_chunk.array for frame in all_gen_frames if frame.audio_chunk]
    )
    print(
        f"audio_output.shape: {audio_output.shape}, duration: {len(audio_output) / sr} seconds"
    )
    write_video_with_audio(
        "test_streaming.mp4",
        [frame.rgb_image for frame in all_gen_frames],
        int16_to_float32(audio_output),
        sr,
        fps=25,
    )

# %%
