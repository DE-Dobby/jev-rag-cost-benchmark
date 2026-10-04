from __future__ import annotations

from .schema import LabelSpan


def is_hallucination_span(span: LabelSpan, *, implicit_true_as_hallucination: bool, due_to_null_as_hallucination: bool) -> bool:
    """RAGTruth의 labels 항목 하나가 "환각으로 셀지" 판단.

    labels 리스트의 모든 항목은 애초에 사람이 식별한 환각 span이지만,
    implicit_true(사실은 맞지만 근거 문서에 없는 정보)와 due_to_null(구조화 데이터의
    null 값 때문에 생긴 환각)은 정책에 따라 "환각"으로 셀지 말지가 갈린다.
    """
    if span.implicit_true and not implicit_true_as_hallucination:
        return False
    if span.due_to_null and not due_to_null_as_hallucination:
        return False
    return True


def is_hallucinated_response(
    labels: list[LabelSpan],
    *,
    implicit_true_as_hallucination: bool,
    due_to_null_as_hallucination: bool,
) -> bool:
    """응답 단위 라벨: 필터링 후에도 남는 환각 span이 하나라도 있으면 1."""
    return any(
        is_hallucination_span(
            span,
            implicit_true_as_hallucination=implicit_true_as_hallucination,
            due_to_null_as_hallucination=due_to_null_as_hallucination,
        )
        for span in labels
    )
