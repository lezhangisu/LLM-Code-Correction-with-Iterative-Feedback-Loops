"""
config_loader.py
================

Loads API keys and configuration from a YAML file.
The config file should be placed at config/api_keys.yaml (relative to project root).
"""

import os
import yaml

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "api_keys.yaml",
)

_config_cache = None


def load_config(config_path=None):
    """Load API keys configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file.
                     Defaults to config/api_keys.yaml in project root.

    Returns:
        dict: Configuration dictionary with provider keys.
    """
    global _config_cache
    if _config_cache is not None and config_path is None:
        return _config_cache

    path = config_path or _DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Please copy config/api_keys.yaml.example to config/api_keys.yaml "
            f"and fill in your API keys."
        )
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if config_path is None:
        _config_cache = config
    return config


def get_provider_config(provider_name, config_path=None):
    """Get API key and base_url for a specific provider.

    Args:
        provider_name: Key in the config file (e.g., 'deepseek_official', 'openai').
        config_path: Optional path override.

    Returns:
        tuple: (api_key, base_url)
    """
    config = load_config(config_path)
    if provider_name not in config:
        raise KeyError(
            f"Provider '{provider_name}' not found in config. "
            f"Available providers: {list(config.keys())}"
        )
    provider = config[provider_name]
    api_key = provider.get("api_key", "")
    base_url = provider.get("base_url", None)
    if not api_key or api_key.startswith("YOUR_"):
        raise ValueError(
            f"Please set a valid API key for provider '{provider_name}' "
            f"in config/api_keys.yaml"
        )
    return api_key, base_url


def get_config_section(section_name, config_path=None):
    """Get the full configuration dict for a section.

    Args:
        section_name: Key in the config file (e.g., 'leetcode', 'openai').
        config_path: Optional path override.

    Returns:
        dict: Configuration dictionary for the section
    """
    config = load_config(config_path)
    if section_name not in config:
        raise KeyError(
            f"Section '{section_name}' not found in config. "
            f"Available sections: {list(config.keys())}"
        )
    return config[section_name]
