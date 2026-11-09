"""Score a list of model strings with the contract validator."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from schemasmith.contract.validator import Scorecard, score_outputs
from schemasmith.data.records import TicketRecord

GenerateFn = Callable[[Sequence[str]], list[str]]


def evaluate_records(
    records: Sequence[TicketRecord],
    outputs: Sequence[str] | None = None,
    generate: GenerateFn | None = None,
) -> Scorecard:
    """Score ``outputs`` or the strings returned by ``generate``.

    ``generate`` receives the ticket texts in order and must return one
    string per ticket. This function does not download a model.
    """
    if (outputs is None) == (generate is None):
        raise ValueError("pass outputs or generate, not both")
    texts = [record.text for record in records]
    if generate is not None:
        produced = list(generate(texts))
        if len(produced) != len(records):
            raise ValueError("generate returned a different number of strings than records")
        outputs = produced
    assert outputs is not None
    return score_outputs(
        list(outputs),
        [record.gold for record in records],
        texts,
        [record.language for record in records],
    )
