## Setup

```bash
pip install -r requirements.txt

export BITHUMAN_AVATAR_MODEL="/path/to/model.imx"

# require only one of token or api_secret
# if api_secret is provided, it will fetch temporal token periodically from our API
export BITHUMAN_RUNTIME_TOKEN=your-bithuman-token
export BITHUMAN_API_SECRET=your-api-secret

export OPENAI_API_KEY=your-openai-token
```

## Run Agent Locally
```bash
python agent_local.py
```

## Run Agent using LiveKit WebRTC
```bash
export LIVEKIT_URL=wss://...
export LIVEKIT_API_KEY=devkey
export LIVEKIT_API_SECRET=secret

python agent_webrtc.py dev
```

