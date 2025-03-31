import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from bithuman.audio import get_mel_chunks, load_audio, write_video_with_audio
from bithuman.config import load_settings
from bithuman.lib.generator import BithumanGenerator, CompressionType

root = Path(
    "/Volumes/razer-data/cpu-asserts/Dec10/workspaces_720p/Gen-3 Alpha Turbo 195528817, Cropped - Sunflower_, M 5"
)
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJiaXRodW1hbiIsImlhdCI6MTc0MDg0MjY1NiwiZXhwIjoxNzQwODQ2MjU2fQ.OtdEwUkr5L1DVZpHqKo1KDWJ9dVlXp4fs_S4bNwsW4E"


def test_generator():
    # Initialize generator with model
    settings = load_settings()
    runtime = BithumanGenerator(settings.AUDIO_ENCODER_PATH)
    print(f"Instance ID: {runtime.get_instance_id()}")
    print(f"Validate token: {runtime.validate_token(token)}")

    # Load audio feature clusters
    audio_features = np.load(root / "precompute_videos" / "feature_centers.npy")
    runtime.set_audio_feature(audio_features)

    # Add video
    print("Loading video...")
    runtime.add_video(
        video_name="test_video",
        video_path=str(
            root
            / "videos"
            / "Gen-3 Alpha Turbo 195528817, Cropped - Sunflower_, M 5.mp4"
        ),
        video_data_path=str(
            root
            / "videos"
            / "Gen-3 Alpha Turbo 195528817, Cropped - Sunflower_, M 5.mp4.WAV2LIP_1280_a5f142d7.h5"
        ),
        avatar_data_path=str(
            root
            / "precompute_videos"
            / "/Volumes/razer-data/cpu-asserts/Dec10/workspaces_720p/Gen-3 Alpha Turbo 195528817, Cropped - Sunflower_, M 5/precompute_videos/"
            "Gen-3 Alpha Turbo 195528817, Cropped - Sunflower_, M 5.feature-first.bhtensor"
        ),
        compression_type=CompressionType.JPEG,
    )
    num_frames = runtime.get_num_frames("test_video")
    print(f"Number of frames: {num_frames}")

    # Load audio file
    audio_np, _ = load_audio("python_module/examples/sample.wav", 16000)
    tic = time.time()
    mel_chunks = get_mel_chunks(audio_np, 25)
    toc = time.time()
    print(f"Mel chunks time: {toc - tic} seconds, fps: {len(mel_chunks) / (toc - tic)}")

    # Process each mel chunk
    output_dir = Path("test_outputs/outputs2")
    output_dir.mkdir(exist_ok=True)

    runtimes = []
    frames = []
    frame_index = -1
    inc = 1
    mel_chunks = mel_chunks
    for mel_chunk in tqdm(mel_chunks, total=len(mel_chunks)):
        # Process audio and get frame
        tic = time.time()
        frame_index += inc
        if frame_index >= num_frames:
            inc = -1
            frame_index = num_frames - 1
        elif frame_index < 0:
            inc = 1
            frame_index = 0

        frame = runtime.process_audio(
            mel_chunk=mel_chunk, video_name="test_video", frame_idx=frame_index
        )

        toc = time.time()
        runtimes.append(toc - tic)

        frames.append(frame[..., ::-1])  # BGR to RGB
        # if len(frames) > 100:
        #     break

    # Write video with audio
    write_video_with_audio(
        output_path=output_dir / "output.mp4",
        frames=frames,
        audio_np=audio_np,
        sample_rate=16000,
        fps=25,
    )

    print("Test completed successfully!")
    print(f"Average runtime: {np.mean(runtimes) * 1000} ms")


if __name__ == "__main__":
    test_generator()
