# bitHuman SDK Examples

This repository contains examples demonstrating how to use the bitHuman SDK to create interactive digital agents that respond to audio input. These examples showcase different ways to deploy and interact with bitHuman rendering SDK.

## Prerequisites

**Supported Python Versions:**
- Python 3.10 to 3.13

**Supported Operating Systems:**
- Linux (x86_64 and arm64)
- macOS (Apple Silicon, macOS >= 15)

## Installation

1. Install the bitHuman SDK:
```bash
pip install bithuman
```

2. Install additional dependencies based on the example you want to run (see the README in each example folder).

## Getting Started

### Obtaining an API Secret Key

You need a bitHuman API secret key to use these examples:

1. Visit [bitHuman ImagineX](https://console.bithuman.io/imagineX) to sign up
2. Create a new API secret key
3. Set your API secret key in your config file

```bash
export BITHUMAN_API_SECRET='your_api_secret'
```

### Models

You'll need a bitHuman imagineX model (`.imx` file) to run these examples. These models define the appearance and behavior of your virtual avatar. You can download example models from [bitHuman docs](https://docs.bithuman.io/api-reference/sdk/quick-start).

Set the path to your avatar model:
```bash
export BITHUMAN_AVATAR_MODEL='/path/to/model/avatar.imx'
```


## Examples Overview

### 1. Avatar Echo

Basic example that captures audio from your microphone, processes it with the bitHuman SDK, and displays the animated avatar in a local window.

```bash
python avatar/echo.py
```

### 2. LiveKit WebRTC Integration

Stream a bitHuman avatar to a LiveKit room using WebRTC, while controlling the avatar's speech through a WebSocket interface.

```bash
# Start the server
python livekit_webrtc/bithuman_server.py --room test

# Send audio to the avatar
python livekit_webrtc/websocket_client.py stream /path/audio.wav
```

### 3. LiveKit Agent

Run an voice agent with bitHuman rendering capabilities:

```bash
# Run locally
python livekit_agent/agent_local.py

# Run in a LiveKit room
python livekit_agent/agent_webrtc.py dev
```

### 4. FastRTC

Run a LiveKit agent with FastRTC WebRTC implementation:

```bash
python fastrtc/fastrtc_example.py
```

## Directory Structure

```
sdk-examples-python/
├── README.md              # This file
├── avatar/                # Basic avatar example
├── livekit_webrtc/        # LiveKit WebRTC integration example
├── livekit_agent/         # LiveKit agent examples
└── fastrtc/               # FastRTC WebRTC example
```

## API Overview

The Bithuman Runtime API provides a powerful interface for creating interactive avatars:

### Creating a Runtime Instance

All examples use the bitHuman Runtime (AsyncBithuman) to process audio and generate avatar animations. Here's a basic example of initialization:

```python
from bithuman.runtime import AsyncBithuman
runtime = await AsyncBithuman.create(api_secret="your_api_secret", model_path="/path/to/model.imx")

```

### Core Components

1. **AsyncBithuman**: The main class that handles communication with the Bithuman service.
   - Initialize with your API secret key: `runtime = await AsyncBithuman.create(...)`

   - Interrupt: Cancel ongoing speech with `runtime.interrupt()`

2. **AudioChunk**: Represents audio data for processing.
   - Accepts 16kHz, mono, int16 format audio
   - Can be created from bytes or numpy arrays
   - Provides duration and format conversion utilities

3. **VideoFrame**: Contains the output from the runtime.
   - BGR image data (numpy array)
   - Synchronized audio chunks
   - Frame metadata (index, message ID)


### Input/Output Flow

1. **Input**:
   - Send audio data (16kHz, mono, int16 format) to the runtime
   - The runtime processes this input to generate appropriate avatar animations

2. **Processing**:
   - The runtime analyzes the audio to determine appropriate facial movements and expressions
   - Frames are generated at 25 FPS with synchronized audio

3. **Output**:
   - Each frame contains both visual data (BGR image) and the corresponding audio chunk
   - The example shows how to render these frames and play the audio in real-time


## Additional Resources

- [Bithuman Documentation](https://docs.bithuman.io)
- [LiveKit Agents](https://github.com/livekit/agents)
