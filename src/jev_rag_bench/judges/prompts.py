from __future__ import annotations

# 판정 입력은 (근거 문서, 응답)으로 고정한다. RAGTruth 자체의 사람 주석 기준이
# "응답이 근거 문서에 없는 내용을 말하거나(baseless) 근거와 모순되는가(conflict)"이고,
# 질문/프롬프트가 맞는 답인지는 보지 않기 때문에 우리 판정도 동일한 기준을 따른다.
# (task_type은 source_info 구조를 해석하는 데만 쓰는 메타 정보로 같이 넘긴다)

# Jev Noul 질문 문구. v1이 기본, v2/v3는 문구 민감도 ablation용.
JEV_NOUL_QUESTIONS: dict[str, str] = {
    "v1": (
        "Does the response contain any claim that is not supported by the context, "
        "or that contradicts the context?"
    ),
    "v2": (
        "Is there any statement in the response that the context does not support "
        "or that conflicts with the context?"
    ),
    "v3": (
        "Considering only the context given, does the response assert anything "
        "unsupported by or contradictory to that context?"
    ),
}

JEV_NOUL_CRITERIA: dict[str, str] = {
    "true": "At least one claim in the response is absent from, unsupported by, or contradicts the context.",
    "false": "Every claim in the response is supported by the context.",
}

# A(기준)/폴백/C(대조군) Gemini 호출에 공통으로 쓰는 프롬프트. JEV_NOUL_QUESTIONS["v1"]과
# 같은 기준으로 맞춰 Jev와 Gemini 판정이 같은 질문을 보고 있도록 한다.
GEMINI_JUDGE_PROMPT_TEMPLATE = """You are judging whether an LLM response contains a hallucination \
relative to the given context. A hallucination is any claim in the response that is \
not supported by the context, or that contradicts the context. Do not judge whether \
the response correctly answers any question — judge only whether its claims are \
grounded in the context.

Task type: {task_type}

Context:
{context}

Response:
{response}

Does the response contain any claim that is not supported by the context, or that \
contradicts the context? Answer with the required JSON schema."""


def build_gemini_prompt(*, task_type: str, context: str, response: str) -> str:
    return GEMINI_JUDGE_PROMPT_TEMPLATE.format(task_type=task_type, context=context, response=response)
