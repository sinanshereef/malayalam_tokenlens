"""MalayalamTokenLens MCP server."""
from fastmcp import FastMCP

from . import analysis
from .samples import SAMPLES

mcp = FastMCP(
    "MalayalamTokenLens",
    instructions=("Measures how many tokens Malayalam text costs in different LLM tokenizers compared with English. "
                  "Tokenizer specs: 'tiktoken:o200k_base', 'tiktoken:cl100k_base', 'hf:<repo-id>', 'bytes'."),
)


@mcp.tool()
def count_tokens(text: str, tokenizer: str = "tiktoken:o200k_base") -> dict:
    """Count tokens, words and characters of a text for one tokenizer."""
    return analysis.text_stats(text, tokenizer)


@mcp.tool()
def compare_tokenizers(text: str, tokenizers: list[str] | None = None) -> dict:
    """Compare token counts, tokens-per-word and characters-per-token of one text across several tokenizers."""
    return analysis.compare_tokenizers(text, tokenizers)


@mcp.tool()
def token_tax(malayalam_text: str, english_text: str, tokenizers: list[str] | None = None,
              context_window: int = 8192) -> dict:
    """Given a Malayalam text and its English translation, report how many times more tokens (and API cost)
    Malayalam needs per tokenizer, and how much less fits in a context window."""
    return analysis.token_tax(malayalam_text, english_text, tokenizers, context_window)


@mcp.tool()
def run_sample_benchmark(tokenizers: list[str] | None = None) -> dict:
    """Run token_tax over the built-in Malayalam/English sample sentences and return per-tokenizer totals."""
    ml = "\n".join(m for m, _ in SAMPLES)
    en = "\n".join(e for _, e in SAMPLES)
    result = analysis.token_tax(ml, en, tokenizers)
    result["samples_used"] = len(SAMPLES)
    result["note"] = "Tiny illustrative sample; use token_tax on your own translated text for real conclusions."
    return result


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
