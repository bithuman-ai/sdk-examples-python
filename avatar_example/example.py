import argparse
import asyncio
import os
import sys
from typing import Optional

from dotenv import load_dotenv
from livekit.agents import AgentSession, utils
from livekit.agents.voice.avatar import QueueAudioOutput
from loguru import logger

from bithuman import AsyncBithuman
from bithuman.utils import FPSController
from bithuman.utils.agent import LocalAudioIO, LocalVideoPlayer

load_dotenv()

logger.remove()
logger.add(sys.stdout, level="INFO")


@utils.log_exceptions(logger=logger)
async def read_audio_from_microphone(
    runtime: AsyncBithuman, audio_io: LocalAudioIO
) -> None:
    while True:
        if audio_io._agent.input.audio is None:
            await asyncio.sleep(0.1)
            continue

        logger.info("Audio input ready")
        sample_rate = 24000
        audio_stream = utils.audio.AudioByteStream(
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=sample_rate // 100,
        )
        async for frame in audio_io._agent.input.audio:
            for f in audio_stream.push(frame.data):
                await runtime.push_audio(bytes(f.data), f.sample_rate, last_chunk=False)

        await runtime.flush()


async def run_bithuman(
    runtime: AsyncBithuman, audio_file: Optional[str] = None
) -> None:
    """Run the Bithuman runtime with audio and video players."""
    # Initialize Video Player and Audio Input
    audio_io = LocalAudioIO(AgentSession(), QueueAudioOutput(), buffer_size=3)
    await audio_io.start()
    push_audio_task = asyncio.create_task(read_audio_from_microphone(runtime, audio_io))

    video_player = LocalVideoPlayer(window_size=runtime.get_frame_size(), buffer_size=3)
    try:
        fps_controller = FPSController(target_fps=25)
        async for frame in runtime.run(out_buffer_empty=video_player.buffer_empty):
            sleep_time = fps_controller.wait_next_frame(sleep=False)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

            if frame.has_image:
                await video_player.capture_frame(
                    frame,
                    fps=fps_controller.average_fps,
                    exp_time=runtime.get_expiration_time(),
                )

            if frame.audio_chunk is not None:
                await audio_io.capture_frame(frame.audio_chunk)

            fps_controller.update()

    except asyncio.CancelledError:
        logger.info("Runtime task cancelled")
    finally:
        # Clean up
        if push_audio_task and not push_audio_task.done():
            push_audio_task.cancel()
        await video_player.aclose()
        await audio_io.aclose()
        await runtime.stop()


async def main(args: argparse.Namespace) -> None:
    """Main entry point for the example application.

    This function demonstrates the proper way to initialize and use the
    AsyncBithuman runtime in an asynchronous context.

    Args:
        args: Command line arguments parsed by argparse.
    """
    logger.info(f"Initializing AsyncBithuman with model: {args.model}")

    # Use the factory method to create a fully initialized instance
    runtime = await AsyncBithuman.create(
        model_path=args.model,
        token=args.token,
        api_secret=args.api_secret,
        insecure=args.insecure,
    )

    # Verify model was set successfully
    try:
        frame_size = runtime.get_frame_size()
        logger.info(f"Model initialized successfully, frame size: {frame_size}")
    except Exception as e:
        logger.error(f"Model initialization verification failed: {e}")
        raise

    # Run the application with the main business logic
    logger.info("Starting runtime...")
    await run_bithuman(runtime, args.audio_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default=os.environ.get("BITHUMAN_AVATAR_MODEL")
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("BITHUMAN_RUNTIME_TOKEN"),
        help="JWT token for authentication (optional if --api-secret is provided)",
    )
    parser.add_argument("--audio-file", type=str, default=None)
    parser.add_argument(
        "--api-secret",
        type=str,
        default=os.environ.get("BITHUMAN_API_SECRET"),
        help="API Secret for API authentication (optional if --token is provided)",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable SSL certificate verification (not recommended for production use)",  # noqa: E501
    )

    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
