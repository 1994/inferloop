"""Model package settings shared by the serving benchmark tools."""

import json
from pathlib import Path


def eos_tokens(model):
    """Match the model package's generation defaults for EOS token precedence."""
    root = Path(model)
    configs = []
    for name in ("generation_config.json", "config.json"):
        path = root / name
        configs.append(json.loads(path.read_text()) if path.exists() else {})
    generation, config = configs
    if not isinstance(generation, dict):
        generation = {}
    if not isinstance(config, dict):
        config = {}
    config = config.get("text_config", config)
    if not isinstance(config, dict):
        config = {}
    if "eos_token_id" in generation:
        value = generation["eos_token_id"]
    elif "eos_token_id" in config:
        value = config["eos_token_id"]
    else:
        return []
    values = value if isinstance(value, list) else [value]
    if any(type(token) is not int or not 0 <= token <= 0xFFFFFFFF for token in values):
        raise ValueError("invalid EOS token")
    return values
