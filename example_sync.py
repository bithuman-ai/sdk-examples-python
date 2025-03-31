import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from bithuman import AudioChunk, Bithuman, VideoFrame
from bithuman.audio import float32_to_int16, load_audio
from bithuman.token_utils import (
    TokenRequestConfig,
    request_token_sync,
    token_refresh_worker_sync,
)
from bithuman.utils import FPSController

logger.remove()
logger.add(sys.stdout, level="INFO")


# simulate video control like push audio/interrupt from another thread
class Controller:
    @dataclass
    class Message:
        audio_file: str = ""
        interrupt: bool = False

    def __init__(self, daemon: Bithuman) -> None:
        self._msg_queue = Queue[Controller.Message | None]()
        self._daemon = daemon

    def push(self, msg: Message):
        self._msg_queue.put(msg)

    def run(self):
        while True:
            msg = self._msg_queue.get()
            if msg is None:
                break
            if msg.interrupt:
                self._daemon.interrupt()
            elif msg.audio_file:
                self._push_audio(msg.audio_file)

    def _push_audio(self, audio_file: str):
        audio_np, sr = load_audio(audio_file)
        audio_np = float32_to_int16(audio_np)

        # simulate streaming audio bytes
        chunk_size = sr // 100
        for i in range(0, len(audio_np), chunk_size):
            chunk = audio_np[i : i + chunk_size]
            self._daemon.push_audio(chunk.tobytes(), sr, last_chunk=False)
        # flush the audio, mark the end of speech
        self._daemon.flush()


def render_image(
    frame: VideoFrame, fps_controller: FPSController, start_time: float, exp_time: float
) -> np.ndarray:
    image = frame.bgr_image
    scale = 720 / max(image.shape)
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale)
    fps = fps_controller.fps
    elapsed_time = time.time() - start_time
    cv2.putText(
        image,
        f"FPS: {fps:.1f} Elapsed: {elapsed_time:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        lineType=cv2.LINE_AA,
    )
    if exp_time > 0:
        left_time = exp_time - time.time()
        if left_time < 3600:
            cv2.putText(
                image,
                f"Expiration: {left_time:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                lineType=cv2.LINE_AA,
            )
    return image


def request_token(runtime: Bithuman, args: argparse.Namespace) -> Optional[str]:
    """Request a token from the API using the provided arguments."""
    config = TokenRequestConfig(
        api_url=args.api_url,
        api_secret=args.api_secret,
        tags=args.tags,
        insecure=args.insecure,
    )
    return request_token_sync(runtime, config)


def token_refresh_worker(runtime: Bithuman, args: argparse.Namespace, stop_event: threading.Event):
    """Worker thread that refreshes the token every minute."""
    config = TokenRequestConfig(
        api_url=args.api_url,
        api_secret=args.api_secret,
        fingerprint=runtime.fingerprint,
        tags=args.tags,
        insecure=args.insecure,
    )
    
    def on_token_refresh(token: str) -> None:
        runtime._token = token
        try:
            runtime.set_token(token)
            logger.debug("Token refreshed and set successfully")
        except Exception as e:
            logger.error(f"Error setting refreshed token: {e}")
    
    token_refresh_worker_sync(
        config, 
        stop_event, 
        on_token_refresh=on_token_refresh
    )


def main(args: argparse.Namespace):
    # Use the factory method to create a fully initialized instance
    logger.info(f"Creating Bithuman instance with model: {args.model}")
    
    try:
        runtime = Bithuman.create(
            model_path=args.model,
            token=args.token,
            api_secret=args.api_secret,
            insecure=args.insecure
        )
        logger.info("Bithuman instance created successfully")
    except Exception as e:
        logger.error(f"Failed to create Bithuman instance: {e}")
        sys.exit(1)
    
    controller = Controller(runtime)
    controller_thread = threading.Thread(target=controller.run)
    controller_thread.start()

    fps_controller = FPSController(target_fps=25)

    start_time: Optional[float] = None
    for frame in runtime.run():
        if not frame.has_image:
            # metadata frame
            logger.info(f"Received metadata frame: {frame}")
            continue

        if start_time is None:
            logger.info("Streaming started")
            start_time = time.time()

        # audio and the generated image
        frame: VideoFrame
        image: np.ndarray = frame.bgr_image
        audio: AudioChunk | None = frame.audio_chunk  # noqa: F841
        exp_time: float = runtime.get_expiration_time()

        # consume the video and audio, here just show the image in opencv
        image = render_image(frame, fps_controller, start_time, exp_time)

        fps_controller.wait_next_frame()

        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("1"):
            logger.info(f"Pushing audio file: {args.audio_file}")
            controller.push(Controller.Message(audio_file=args.audio_file))
        elif key == ord("2"):
            logger.info("Interrupting")
            controller.push(Controller.Message(interrupt=True))
        elif key == ord("q"):
            logger.info("Exiting...")
            controller.push(None)
            break

        fps_controller.update()

        logger.info("Stopping controller thread...")
        controller.push(None)
        controller_thread.join(timeout=2)

        logger.info("Cleanup complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
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
        help="Skip SSL verification for API requests",
    )
    args = parser.parse_args()

    main(args)
