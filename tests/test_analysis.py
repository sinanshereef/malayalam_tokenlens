import pytest

from malayalam_tokenlens import analysis
from malayalam_tokenlens.samples import SAMPLES
from malayalam_tokenlens.tokenizers import TokenizerError, count_tokens

ML, EN = SAMPLES[0]


def test_bytes_baseline_counts_utf8():
    assert count_tokens("abc", "bytes") == 3
    assert count_tokens("അ", "bytes") == 3  # one Malayalam letter = 3 UTF-8 bytes


def test_malayalam_ratio():
    assert analysis.malayalam_ratio(ML) > 0.9
    assert analysis.malayalam_ratio(EN) == 0.0


def test_token_tax_bytes_malayalam_costs_more():
    row = analysis.token_tax(ML, EN, ["bytes"])["results"][0]
    assert row["token_tax"] > 1
    assert row["messages_fitting_context"]["malayalam"] < row["messages_fitting_context"]["english"]


def test_warns_when_text_is_not_malayalam():
    assert analysis.token_tax(EN, EN, ["bytes"])["warnings"]


def test_bad_tokenizer_is_reported_not_fatal():
    out = analysis.compare_tokenizers(ML, ["bytes", "nonsense"])
    assert "tokens" in out["results"][0] and "error" in out["results"][1]


def test_unknown_spec_raises():
    with pytest.raises(TokenizerError):
        count_tokens("x", "nonsense")


def test_empty_text_rejected():
    with pytest.raises(ValueError):
        analysis.text_stats("  ", "bytes")


def test_tiktoken_if_available():
    try:
        import tiktoken
        tiktoken.get_encoding("o200k_base")
    except Exception:
        pytest.skip("tiktoken vocabulary not downloadable here")
    assert analysis.token_tax(ML, EN, ["tiktoken:o200k_base"])["results"][0]["token_tax"] > 1
