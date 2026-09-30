"""Tokenizer backends. A tokenizer is named by a spec string:

    tiktoken:o200k_base   OpenAI tokenizer (GPT-4o family)
    tiktoken:cl100k_base  OpenAI tokenizer (GPT-4 family)
    hf:<repo-id>          any Hugging Face tokenizer, e.g. hf:Qwen/Qwen2.5-0.5B  (needs the [hf] extra)
    bytes                 UTF-8 byte count, a worst-case baseline that works offline
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable

DEFAULT_TOKENIZERS = ["tiktoken:o200k_base", "tiktoken:cl100k_base", "bytes"]


class TokenizerError(ValueError):
    """Raised when a tokenizer spec is invalid or cannot be loaded."""


@lru_cache(maxsize=32)
def _load(spec: str) -> Callable[[str], int]:
    kind, _, name = spec.partition(":")
    try:
        if spec == "bytes":
            return lambda text: len(text.encode("utf-8"))
        if kind == "tiktoken" and name:
            import tiktoken
            enc = tiktoken.get_encoding(name)
            return lambda text: len(enc.encode(text))
        if kind == "hf" and name:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_pretrained(name)
            return lambda text: len(tok.encode(text, add_special_tokens=False).ids)
    except Exception as exc:  # network failure, unknown encoding, missing extra...
        raise TokenizerError(f"Could not load '{spec}': {type(exc).__name__}: {exc}") from exc
    raise TokenizerError(
        f"Unknown tokenizer '{spec}'. Use 'tiktoken:<encoding>', 'hf:<repo-id>' or 'bytes'.")


def count_tokens(text: str, spec: str) -> int:
    return _load(spec)(text)
