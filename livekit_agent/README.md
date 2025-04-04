## Run Agent with Local Audio and Video IO

```bash
pip install -r requirements.txt

export BITHUMAN_AVATAR_MODEL="/path/to/model.imx"

# require only one of token or api_secret
# if api_secret is provided, it will fetch temporal token periodically from our API
export BITHUMAN_RUNTIME_TOKEN="your-token"
export BITHUMAN_API_SECRET="your-api-secret"
python local_agent.py
```