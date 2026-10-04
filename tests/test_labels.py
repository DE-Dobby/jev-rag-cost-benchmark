from jev_rag_bench.data.labels import is_hallucinated_response
from jev_rag_bench.data.schema import LabelSpan


def _span(*, implicit_true: bool, due_to_null: bool) -> LabelSpan:
    return LabelSpan(
        start=0,
        end=1,
        text="x",
        meta=None,
        label_type="Evident Conflict",
        implicit_true=implicit_true,
        due_to_null=due_to_null,
    )


def test_no_labels_is_not_hallucinated():
    assert is_hallucinated_response(
        [], implicit_true_as_hallucination=True, due_to_null_as_hallucination=False
    ) is False


def test_plain_label_is_hallucinated():
    spans = [_span(implicit_true=False, due_to_null=False)]
    assert is_hallucinated_response(
        spans, implicit_true_as_hallucination=True, due_to_null_as_hallucination=False
    ) is True


def test_due_to_null_excluded_by_default():
    spans = [_span(implicit_true=False, due_to_null=True)]
    assert is_hallucinated_response(
        spans, implicit_true_as_hallucination=True, due_to_null_as_hallucination=False
    ) is False


def test_due_to_null_included_when_configured():
    spans = [_span(implicit_true=False, due_to_null=True)]
    assert is_hallucinated_response(
        spans, implicit_true_as_hallucination=True, due_to_null_as_hallucination=True
    ) is True


def test_implicit_true_excluded_when_configured_off():
    spans = [_span(implicit_true=True, due_to_null=False)]
    assert is_hallucinated_response(
        spans, implicit_true_as_hallucination=False, due_to_null_as_hallucination=False
    ) is False


def test_one_qualifying_span_among_many_is_enough():
    spans = [
        _span(implicit_true=False, due_to_null=True),  # 필터링되어 제외
        _span(implicit_true=False, due_to_null=False),  # 이 하나로 환각 확정
    ]
    assert is_hallucinated_response(
        spans, implicit_true_as_hallucination=True, due_to_null_as_hallucination=False
    ) is True
