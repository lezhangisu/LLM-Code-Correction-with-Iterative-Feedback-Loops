"""
leetcode_api package
====================

Core package for the iterative feedback loop benchmark.

Provides:
  - LeetcodeAPI: Interact with leetcode.com (fetch problems, submit code, get results)
  - Model: LLM API wrappers for various providers (Deepseek, OpenAI, Qwen, Llama, Claude)
  - ResultAnalyzer: Collect and merge experimental results into DataFrames/CSV
  - utils: Helper functions for data extraction, file I/O, and text processing
"""

from .leetcode import LeetcodeAPI
from .analyzer import ResultAnalyzer
from .models import Model
from .config_loader import load_config, get_provider_config
from .utils import *
