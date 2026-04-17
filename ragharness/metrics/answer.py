"""Answer-quality metrics between strict ``exact_match`` and ``llm_judge``.

All three scorers here are pure-Python with zero dependencies. They
share a single :func:`_tokenize` helper so tokenisation is consistent
across ``f1_token`` and ``rouge_l``.
"""

from __future__ import annotations

import string
from collections import Counter

from ragharness.dataset import EvalItem
from ragharness.protocol import RAGResult

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return text.lower().translate(_PUNCT_TABLE).split()


def contains(item: EvalItem, result: RAGResult) -> float:
    """1.0 if the expected answer appears as a (normalised) substring of the system answer.

    Case-insensitive; surrounding whitespace on both sides is stripped.
    The pragmatic middle ground between the strict :func:`exact_match`
    and the expensive LLM judge — most open-ended QA datasets are scored
    fine with this.
    """
    expected = item.expected_answer.strip().lower()
    answer = result.answer.strip().lower()
    if not expected:
        return 0.0
    return 1.0 if expected in answer else 0.0


def f1_token(item: EvalItem, result: RAGResult) -> float:
    """SQuAD-style token-level F1 between expected and system answer.

    Tokenisation: lowercase, strip ASCII punctuation, split on whitespace.
    Precision and recall are computed against the *multiset* intersection
    (so repeated tokens in one side don't over-credit a single match on
    the other). Returns 0.0 if either side tokenises to empty.
    """
    pred_tokens = _tokenize(result.answer)
    ref_tokens = _tokenize(item.expected_answer)
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence between two token lists."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    curr = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[-1]


def rouge_l(item: EvalItem, result: RAGResult) -> float:
    """ROUGE-L F-measure based on the longest common subsequence of tokens.

    Same tokenisation as :func:`f1_token`. ``P = LCS / |pred|``,
    ``R = LCS / |ref|``, ``F = 2·P·R / (P + R)``. Returns 0.0 if either
    side tokenises to empty or the two share no common subsequence.
    """
    pred_tokens = _tokenize(result.answer)
    ref_tokens = _tokenize(item.expected_answer)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)
