"""
models.py
=========

LLM model wrappers for the iterative feedback loop benchmark.

Supports two experiment modes:
  - Single-attempt (baseline): takes a single prompt string, returns answer text.
  - Iterative (loop): takes a list of chat messages, returns an assistant message list.

All API keys are loaded from config/api_keys.yaml via config_loader.

Models used in the paper:
  - DeepSeek-R1-0528 (via DeepInfra)
  - DeepSeek-V3-0324 (via DeepInfra)
  - GPT-o4-mini (via OpenAI)
  - GPT-4.1-mini (via OpenAI)
"""

import requests
from openai import OpenAI
from .config_loader import get_provider_config


class Model:
    """Provides methods to call LLM APIs for experiments."""

    # ------------------------------------------------------------------
    # Internal streaming helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_client(provider, base_url_override=None):
        """Create an OpenAI-compatible client from config."""
        api_key, base_url = get_provider_config(provider)
        return OpenAI(
            api_key=api_key,
            base_url=base_url_override or base_url,
        )

    def _stream_openai(self, client, model_name, messages,
                       temperature=0.1, top_p=0.95,
                       stream_options=None, extra_body=None,
                       timeout=None, max_tokens=None,
                       has_thinking=False):
        """Stream a chat completion from an OpenAI-compatible endpoint.

        Args:
            client: OpenAI client instance.
            model_name: Model identifier string.
            messages: List of message dicts.
            temperature: Sampling temperature.
            top_p: Nucleus sampling top_p.
            stream_options: Extra stream options (e.g., include_usage).
            extra_body: Extra body parameters (e.g., enable_thinking).
            timeout: Request timeout in seconds.
            max_tokens: Maximum tokens to generate.
            has_thinking: If True, extract reasoning_content from delta.

        Returns:
            tuple: (answer_content_str, prompt_tokens, completion_tokens)
        """
        max_retries = 3
        for retry in range(max_retries):
            answer = ""
            reasoning = ""
            is_answering = False
            prompt_tokens = 0
            completion_tokens = 0
            try:
                kwargs = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stream": True,
                }
                if stream_options:
                    kwargs["stream_options"] = stream_options
                if extra_body:
                    kwargs["extra_body"] = extra_body
                if timeout:
                    kwargs["timeout"] = timeout
                if max_tokens:
                    kwargs["max_tokens"] = max_tokens

                completion = client.chat.completions.create(**kwargs)

                try:
                    for chunk in completion:
                        if not chunk.choices:
                            if chunk.usage:
                                prompt_tokens = chunk.usage.prompt_tokens
                                completion_tokens = chunk.usage.completion_tokens
                            continue

                        delta = chunk.choices[0].delta

                        if chunk.choices[0].finish_reason:
                            if chunk.usage:
                                prompt_tokens = chunk.usage.prompt_tokens
                                completion_tokens = chunk.usage.completion_tokens
                            continue

                        if has_thinking and hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                            print(delta.reasoning_content, end='', flush=True)
                            reasoning += delta.reasoning_content
                            continue

                        if delta.content:
                            if not is_answering:
                                print("\n--- ANSWER ---")
                                is_answering = True
                            print(delta.content, end='', flush=True)
                            answer += delta.content

                    return answer, prompt_tokens, completion_tokens

                except (ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
                    print(f"\nStreaming error: {e}")
                    if retry < max_retries - 1:
                        print(f"Retrying ({retry + 1}/{max_retries})...")
                        continue
                    return answer, prompt_tokens, completion_tokens

            except Exception as e:
                print(f"API error: {e}")
                if retry < max_retries - 1:
                    print(f"Retrying ({retry + 1}/{max_retries})...")
                    continue
                return "", 0, 0

        return "", 0, 0

    def _stream_openai_message_result(self, client, model_name, messages, **kwargs):
        """Wrapper that returns the result as a message list for loop experiments.

        Returns:
            tuple: (message_list, prompt_tokens, completion_tokens)
        """
        answer, pt, ct = self._stream_openai(client, model_name, messages, **kwargs)
        msg = [{"role": "assistant", "content": answer}] if answer else []
        return msg, pt, ct

    # ==================================================================
    # Experiment 1: Single-attempt (baseline) models
    # ==================================================================

    def deepseek_r1_deepinfra(self, prompt, temperature=0.1, top_p=0.95):
        """DeepSeek-R1-0528 via DeepInfra."""
        client = self._make_client("deepinfra")
        return self._stream_openai(
            client, "deepseek-ai/DeepSeek-R1-0528",
            [{"role": "user", "content": prompt}],
            temperature=temperature, top_p=top_p,
            stream_options={"include_usage": True},
            has_thinking=True,
        )

    def deepseek_v3_deepinfra(self, prompt, temperature=0.1, top_p=0.95):
        """DeepSeek-V3-0324 via DeepInfra."""
        client = self._make_client("deepinfra")
        return self._stream_openai(
            client, "deepseek-ai/DeepSeek-V3-0324",
            [{"role": "user", "content": prompt}],
            temperature=temperature, top_p=top_p,
            stream_options={"include_usage": True},
        )

    def gpt_41_mini(self, prompt, temperature=0.1, top_p=0.95):
        """GPT-4.1-mini via OpenAI API."""
        client = self._make_client("openai")
        return self._stream_openai(
            client, "gpt-4.1-mini-2025-04-14",
            [{"role": "user", "content": prompt}],
            temperature=temperature, top_p=top_p,
            stream_options={"include_usage": True},
        )

    # ==================================================================
    # Experiment 2: Iterative (loop) models
    # These accept multi-turn message history for the feedback loop.
    # ==================================================================

    def deepseek_r1_deepinfra_loop_stream(self, messages, temperature=0.1, top_p=0.95):
        """DeepSeek-R1-0528 loop mode via DeepInfra."""
        client = self._make_client("deepinfra")
        return self._stream_openai_message_result(
            client, "deepseek-ai/DeepSeek-R1-0528", messages,
            temperature=temperature, top_p=top_p,
            stream_options={"include_usage": True},
            timeout=1200, has_thinking=True,
        )

    def deepseek_v3_deepinfra_loop_stream(self, messages, temperature=0.1, top_p=0.95):
        """DeepSeek-V3-0324 loop mode via DeepInfra."""
        client = self._make_client("deepinfra")
        return self._stream_openai_message_result(
            client, "deepseek-ai/DeepSeek-V3-0324", messages,
            temperature=temperature, top_p=top_p,
            stream_options={"include_usage": True},
        )

    def gpt_o4_mini_loop(self, messages):
        """GPT o4-mini loop mode via OpenAI. (Temperature/top_p not configurable.)"""
        client = self._make_client("openai")
        return self._stream_openai_message_result(
            client, "o4-mini-2025-04-16", messages,
            stream_options={"include_usage": True},
        )

    def gpt_41_mini_loop(self, messages, temperature=0.3, top_p=1.0):
        """GPT-4.1-mini loop mode via OpenAI."""
        client = self._make_client("openai")
        return self._stream_openai_message_result(
            client, "gpt-4.1-mini-2025-04-14", messages,
            temperature=temperature, top_p=top_p,
            stream_options={"include_usage": True},
        )
