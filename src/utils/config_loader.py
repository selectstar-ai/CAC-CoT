import yaml
import os

def load_config(path: str):
    """Load configuration from a YAML file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_prompts(path: str = None):
    """Load prompts from the standard prompts.yaml location."""
    if path is None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(root, "prompts", "prompts.yaml")
    
    return load_config(path)
