# MalayalamTokenLens MCP

An MCP server that answers one question: **how much more does Malayalam cost in an LLM than English?**

LLMs bill and limit by *tokens*, and tokenizers trained mostly on English split Malayalam script into many
small pieces. The same sentence can cost several times more, and fill a context window several times faster.
Ask any MCP client *"what is the token tax for this Malayalam sentence on GPT-4o?"* and get numbers back.

## Tools

| Tool | What it does |
|---|---|
| `count_tokens` | Tokens, words, characters for one text and one tokenizer |
| `compare_tokenizers` | The same text across several tokenizers (tokens per word, characters per token) |
| `token_tax` | Malayalam text + its English translation gives the cost multiplier and how many messages fit in a context window |
| `run_sample_benchmark` | `token_tax` over built-in sample sentence pairs |

Tokenizers are named by a spec string: `tiktoken:o200k_base` (GPT-4o), `tiktoken:cl100k_base` (GPT-4),
`hf:<repo-id>` (any Hugging Face tokenizer, e.g. `hf:Qwen/Qwen2.5-0.5B`) and `bytes` (UTF-8 bytes, an offline worst-case baseline).
If one tokenizer can't load, the others still return results.

## Install

```bash
pip install -e .            # core (tiktoken tokenizers + bytes)
pip install -e ".[hf]"      # add Hugging Face tokenizers
```

Claude Desktop config:

```json
{ "mcpServers": { "malayalam-tokenlens": { "command": "malayalam-tokenlens-mcp" } } }
```

Claude Code: `claude mcp add malayalam-tokenlens -- malayalam-tokenlens-mcp`

## Try it

> Run token_tax with malayalam_text "ഇന്ന് കേരളത്തിൽ കനത്ത മഴ പെയ്യുന്നു." and english_text
> "It is raining heavily in Kerala today." using tiktoken:o200k_base, tiktoken:cl100k_base and bytes.

The tokenizer vocabularies are downloaded on first use, so the first call needs internet.

## Measure your own numbers

The built-in samples are only four sentence pairs, enough for a demo but **not a benchmark**. For real
conclusions, use a larger, professionally translated set (for example the Malayalam and English sides of FLORES-200)
and report the totals from `token_tax`.

## Tests

```bash
pip install -e ".[dev]" && pytest
```

The tiktoken test is skipped automatically when the vocabulary can't be downloaded.

## Related work and honest scope

Tokenizer efficiency for Indian languages is an established research topic, and generic token-counting MCP
servers exist. This project does not claim a new finding: it puts a Malayalam-specific, translation-paired
measurement (the "token tax") behind an MCP interface, so it can be asked in plain language from any client.

## Limitations

- Measures token counts only, not model quality on Malayalam.
- Compares tokenizers you can load locally; closed models with no public tokenizer (e.g. Claude) are not included.
- Token counts for chat-formatted messages include extra template tokens that are not counted here.
