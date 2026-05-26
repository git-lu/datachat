"""Automated benchmark judge (Lynx-style faithfulness grading via Ollama)."""

from __future__ import annotations

import json
import os
import re

from ollama import chat
from pydantic import BaseModel, Field, field_validator

from benchmarks.pandas_truth import eval_case

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama3.1:8b")
USE_LYNX_PROMPT = os.getenv("JUDGE_USE_LYNX_PROMPT", "false").lower() in {"1", "true", "yes"}

LYNX_STYLE_TEMPLATE = """Given the following QUESTION, DOCUMENT and ANSWER you must analyze the provided answer and determine whether it is faithful to the contents of the DOCUMENT.
The ANSWER must not offer new information beyond the context provided in the DOCUMENT.
The ANSWER must not contradict information provided in the DOCUMENT.
Output your final verdict by strictly following this format: "PASS" if the answer is faithful to the DOCUMENT and "FAIL" if the answer is not faithful to the DOCUMENT.
Show your reasoning.

--
QUESTION (THIS DOES NOT COUNT AS BACKGROUND INFORMATION):
{question}

--
DOCUMENT:
{document}

--
ANSWER:
{answer}

--
Your output should be in JSON FORMAT with the keys "REASONING" and "SCORE":
{{"REASONING": "<your reasoning>", "SCORE": "<PASS or FAIL>"}}"""

IRIS_JUDGE_TEMPLATE = """You grade Iris dataset chatbot answers for factual correctness.

Use the DOCUMENT (expected answer, pandas-computed values, and agent tool output).
The ANSWER is faithful if it matches the facts in the DOCUMENT and does not invent columns or numbers.

Scoring:
- PASS — factually correct and supported by the DOCUMENT
- PARTIAL — mostly right but missing a key fact, wrong precision, or incomplete
- FAIL — wrong numbers, contradicts DOCUMENT, invents columns/data, or ignores tool evidence

Hallucination cases: PASS if the answer clearly states the requested field is not in the dataset.

Output JSON only:
{{"reasoning": "<short bullets>", "score": "PASS" or "PARTIAL" or "FAIL"}}

QUESTION:
{question}

DOCUMENT:
{document}

ANSWER:
{answer}"""


class JudgeVerdict(BaseModel):
    reasoning: str = ""
    score: str = Field(description="PASS, PARTIAL, or FAIL")

    @field_validator("score", mode="before")
    @classmethod
    def _normalize_score(cls, value: object) -> str:
        if value is None:
            return "FAIL"
        text = str(value).strip().upper()
        if text in {"PASS", "PARTIAL", "FAIL"}:
            return text
        if "PARTIAL" in text:
            return "PARTIAL"
        if "PASS" in text:
            return "PASS"
        return "FAIL"

    @property
    def human_label(self) -> str:
        return {"PASS": "correct", "PARTIAL": "partial", "FAIL": "wrong"}[self.score]


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def _pandas_reference(case: dict) -> str:
    code = case.get("pandas")
    if not code:
        return "(no pandas snippet in case)"
    try:
        result = eval_case(case)
        return f"{code}\n→ {result!r}"
    except Exception as exc:
        return f"{code}\n→ error: {exc}"


def build_document(case: dict, result: dict) -> str:
    tool_trace = "\n\n".join(result.get("tool_trace") or []) or "(no tool output)"
    return (
        f"EXPECTED REFERENCE:\n{case.get('expected', '')}\n\n"
        f"PANDAS GROUND TRUTH:\n{_pandas_reference(case)}\n\n"
        f"AGENT TOOL EVIDENCE:\n{tool_trace}\n\n"
        f"AGENT DRAFT:\n{result.get('draft', '')}"
    )


def judge_result(case: dict, result: dict, *, verbose: bool = False) -> dict:
    """Run the judge model on one benchmark result."""
    document = build_document(case, result)
    answer = result.get("answer") or ""

    if USE_LYNX_PROMPT:
        prompt = LYNX_STYLE_TEMPLATE.format(
            question=case["question"],
            document=document,
            answer=answer,
        )
        messages = [{"role": "user", "content": prompt}]
    else:
        prompt = IRIS_JUDGE_TEMPLATE.format(
            question=case["question"],
            document=document,
            answer=answer,
        )
        messages = [{"role": "user", "content": prompt}]

    if verbose:
        print("--- judge prompt ---")
        print(prompt[:2000])
        print("--- end judge prompt ---\n")

    response = chat(
        model=JUDGE_MODEL,
        messages=messages,
        options={"temperature": 0.0, "num_predict": 300},
    )
    raw = response.message.content or ""

    payload = _extract_json(raw)
    if USE_LYNX_PROMPT and "REASONING" in payload:
        payload = {
            "reasoning": payload.get("REASONING", ""),
            "score": payload.get("SCORE", "FAIL"),
        }

    verdict = JudgeVerdict.model_validate(payload)

    return {
        "judge_model": JUDGE_MODEL,
        "judge_score": verdict.score,
        "judge_label": verdict.human_label,
        "judge_reasoning": verdict.reasoning,
        "judge_raw": raw,
    }
