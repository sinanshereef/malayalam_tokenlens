"""Pure analysis logic (no MCP code), so it is easy to test."""
from __future__ import annotations

from .tokenizers import DEFAULT_TOKENIZERS, TokenizerError, count_tokens

MALAYALAM_RANGE = (0x0D00, 0x0D7F)


def malayalam_ratio(text: str) -> float:
    """Share of letters/characters (ignoring whitespace and punctuation) that are Malayalam script."""
    chars = [c for c in text if c.isalpha() or MALAYALAM_RANGE[0] <= ord(c) <= MALAYALAM_RANGE[1]]
    if not chars:
        return 0.0
    return sum(MALAYALAM_RANGE[0] <= ord(c) <= MALAYALAM_RANGE[1] for c in chars) / len(chars)


def text_stats(text: str, spec: str) -> dict:
    if not text.strip():
        raise ValueError("Text is empty.")
    tokens, chars, words = count_tokens(text, spec), len(text), len(text.split())
    return {
        "tokenizer": spec, "tokens": tokens, "characters": chars, "words": words,
        "tokens_per_word": round(tokens / max(words, 1), 2),
        "chars_per_token": round(chars / max(tokens, 1), 2),
    }


def compare_tokenizers(text: str, tokenizers: list[str] | None = None) -> dict:
    """Token statistics for one text across several tokenizers. One failing tokenizer does not stop the rest."""
    rows = []
    for spec in tokenizers or DEFAULT_TOKENIZERS:
        try:
            rows.append(text_stats(text, spec))
        except TokenizerError as exc:
            rows.append({"tokenizer": spec, "error": str(exc)})
    return {"malayalam_ratio": round(malayalam_ratio(text), 2), "results": rows}


def token_tax(malayalam_text: str, english_text: str, tokenizers: list[str] | None = None,
              context_window: int = 8192) -> dict:
    """How many times more tokens the Malayalam text needs than its English equivalent."""
    ratio = malayalam_ratio(malayalam_text)
    warnings = []
    if ratio < 0.5:
        warnings.append(f"Only {ratio:.0%} of the 'Malayalam' text is Malayalam script; results may be misleading.")
    rows = []
    for spec in tokenizers or DEFAULT_TOKENIZERS:
        try:
            ml, en = count_tokens(malayalam_text, spec), count_tokens(english_text, spec)
        except TokenizerError as exc:
            rows.append({"tokenizer": spec, "error": str(exc)})
            continue
        tax = ml / max(en, 1)
        rows.append({
            "tokenizer": spec, "malayalam_tokens": ml, "english_tokens": en,
            "token_tax": round(tax, 2),
            "messages_fitting_context": {"malayalam": context_window // max(ml, 1),
                                         "english": context_window // max(en, 1)},
            "reading": f"Malayalam costs {tax:.1f}x the tokens (and API cost) of the same sentence in English; "
                       f"a {context_window}-token window holds {1 / tax:.0%} as much.",
        })
    return {"malayalam_ratio": round(ratio, 2), "warnings": warnings, "results": rows}
