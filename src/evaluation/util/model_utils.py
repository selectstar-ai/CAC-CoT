import os
import sys

# Find root to load config
def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

sys.path.append(get_project_root())

from src.utils.config_loader import load_config, load_prompts

_root = get_project_root()
_prompts_data = load_prompts()
_models_data = load_config(os.path.join(_root, "configs", "models.yaml"))

SYSTEM_PROMPT = _prompts_data.get("models", {})
MODEL_TO_NAME = _models_data

