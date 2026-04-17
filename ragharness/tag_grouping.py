"""Tag-based metric grouping for per-question scores.

Groups already-computed per-question scores by the ``tags`` dict on each
:class:`~ragharness.dataset.EvalItem`. Items without tags are silently
skipped. Tag keys are discovered dynamically from the data — no config
needed.
"""

from __future__ import annotations

from collections import defaultdict

from ragharness.dataset import EvalItem


def compute_tag_scores(
    items: list[EvalItem],
    per_question_scores: list[dict[str, float]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Group per-question scores by tag key/value and compute means.

    Returns ``{tag_key: {tag_value: {metric_name: mean_score}}}``.

    Example::

        {"topic": {"physics": {"exact_match": 0.6, "f1_token": 0.8},
                   "history": {"exact_match": 0.9, "f1_token": 0.95}}}

    Items with ``tags=None`` or missing a particular tag key are excluded
    from that key's grouping (not counted as "untagged").
    """
    # {tag_key: {tag_value: {metric: [scores...]}}}
    buckets: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for item, scores in zip(items, per_question_scores):
        if not item.tags:
            continue
        for tag_key, tag_value in item.tags.items():
            tag_val_str = str(tag_value)
            for metric_name, score in scores.items():
                buckets[tag_key][tag_val_str][metric_name].append(score)

    result: dict[str, dict[str, dict[str, float]]] = {}
    for tag_key in sorted(buckets):
        result[tag_key] = {}
        for tag_value in sorted(buckets[tag_key]):
            result[tag_key][tag_value] = {
                metric: sum(vals) / len(vals)
                for metric, vals in sorted(buckets[tag_key][tag_value].items())
            }

    return result
