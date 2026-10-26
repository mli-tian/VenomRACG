import os

# LLM-judge configuration. Provide credentials through the environment
# instead of hardcoding them: DEEPSEEK_API_KEY (required) and
# DEEPSEEK_BASE_URL (optional, defaults to the official endpoint).
models_config = {
    "deepseek-flash": {
        "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "key": os.environ.get("DEEPSEEK_API_KEY", "<your_api_key>"),
    },
}
