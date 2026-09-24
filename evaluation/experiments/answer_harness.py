"""Common observable answer harness for controlled evaluation conditions."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from loguru import logger


ANSWER_PROMPT_VERSION = "mosaic-fixed-reader-v2"
ANSWER_SYSTEM_PROMPT = """You are an evaluation assistant answering questions about a user's memory.

Rules:
1. Use only the supplied memory evidence. Do not use outside knowledge.
2. Inspect every attached image, audio recording, or video carefully. Important details may be present only in the native media.
3. If memories conflict or update one another, use the most recent applicable evidence.
4. Return only the shortest concrete answer: a phrase, number, name, or date.
5. Do not provide reasoning, explanation, citations, or hedging.
6. If the evidence is insufficient, return exactly: I don't know."""

KU_ANSWER_SYSTEM_PROMPT = """You are an evaluation assistant answering questions about a user's memory.

Rules:
1. Use only the supplied memory evidence. Do not use outside knowledge.
2. Inspect every attached image, audio recording, or video carefully. Important details may be present only in the native media.
3. For knowledge_update questions (what changed / was updated), answer with the full transition as "from <old> to <new>". Do not answer with only the final/current state when both the earlier and later values appear in the evidence.
4. Return only the shortest concrete answer: a phrase, number, name, or date.
5. Do not provide reasoning, explanation, citations, or hedging.
6. If the evidence is insufficient, return exactly: I don't know."""


@dataclass
class ChatResult:
    """Observable terminal result from one OpenAI-compatible request."""

    text: str = ""
    api_status: str = "empty"
    attempts: int = 0
    finish_reason: str | None = None
    error_class: str | None = None
    error_message: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    latency_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _usage_value(usage: Any, *names: str) -> int | None:
    for name in names:
        if isinstance(usage, Mapping) and usage.get(name) is not None:
            return int(usage[name])
        value = getattr(usage, name, None)
        if value is not None:
            return int(value)
    return None


def _estimate_cost(config: Mapping[str, Any], input_tokens: int | None,
                   output_tokens: int | None) -> float | None:
    input_rate = config.get("input_cost_per_million_tokens")
    output_rate = config.get("output_cost_per_million_tokens")
    if input_rate is None or output_rate is None:
        return None
    return round(
        (input_tokens or 0) * float(input_rate) / 1_000_000
        + (output_tokens or 0) * float(output_rate) / 1_000_000,
        8,
    )


def _system_prompt_for_question_type(question_type: str | None) -> str:
    if str(question_type or "").strip().lower() == "knowledge_update":
        return KU_ANSWER_SYSTEM_PROMPT
    return ANSWER_SYSTEM_PROMPT


def build_answer_messages(
    question: str,
    delivered: Sequence[Mapping[str, Any]],
    media_blocks: Sequence[Mapping[str, Any]],
    question_type: str | None = None,
) -> list[dict[str, Any]]:
    """Build the invariant reader prompt, interleaving native attachments."""
    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": "<MEMORY_EVIDENCE>\n",
    }]
    media_index = 0
    for index, unit in enumerate(delivered, start=1):
        modality = str(unit.get("modality") or "text")
        timestamp = str(unit.get("timestamp") or "unknown time")
        body = str(unit.get("content") or "").strip()
        attachment_note = " | native attachment follows" if unit.get("native_attached") else ""
        content.append({
            "type": "text",
            "text": f"[{index}] {timestamp} | {modality}{attachment_note}\n{body}\n",
        })
        if unit.get("native_attached"):
            if media_index >= len(media_blocks):
                raise ValueError("Delivery manifest has more native units than provider blocks")
            content.append(dict(media_blocks[media_index]))
            media_index += 1
    if media_index != len(media_blocks):
        raise ValueError("Provider blocks contain media not present in the delivery manifest")
    content.append({
        "type": "text",
        "text": f"</MEMORY_EVIDENCE>\n\nQuestion: {question}\nAnswer:",
    })
    return [
        {"role": "system", "content": _system_prompt_for_question_type(question_type)},
        {"role": "user", "content": content},
    ]


class ObservableChatRunner:
    """Retrying OpenAI-compatible caller that retains usage and failures."""

    def __init__(self, config: Mapping[str, Any], client: Any | None = None) -> None:
        self.config = dict(config)
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            from _common import create_openai_client

            self._client = create_openai_client(self.config)
        return self._client

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int | None = None,
    ) -> ChatResult:
        from _common import _apply_extra_body

        started = time.time()
        max_attempts = max(1, int(
            self.config.get("max_attempts", self.config.get("max_retries", 3))
        ))
        backoff = float(self.config.get("retry_backoff", 2.0))
        last_error: Exception | None = None
        last_empty: ChatResult | None = None
        for attempt in range(1, max_attempts + 1):
            kwargs: dict[str, Any] = {
                "model": self.config["llm_model"],
                "messages": list(messages),
                "temperature": float(self.config.get("temperature", 0.0)),
                "max_tokens": int(max_tokens or self.config.get("max_tokens", 512)),
            }
            _apply_extra_body(kwargs, self.config)
            try:
                response = self.client.chat.completions.create(**kwargs)
                choices = getattr(response, "choices", None) or []
                if not choices:
                    raise ValueError("API returned no choices")
                choice = choices[0]
                text = str(getattr(choice.message, "content", "") or "").strip()
                usage = getattr(response, "usage", None)
                input_tokens = _usage_value(usage, "prompt_tokens", "input_tokens")
                output_tokens = _usage_value(usage, "completion_tokens", "output_tokens")
                total_tokens = _usage_value(usage, "total_tokens")
                result = ChatResult(
                    text=text,
                    api_status="ok" if text else "empty",
                    attempts=attempt,
                    finish_reason=getattr(choice, "finish_reason", None),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    estimated_cost_usd=_estimate_cost(
                        self.config, input_tokens, output_tokens
                    ),
                    latency_seconds=round(time.time() - started, 3),
                )
                if text:
                    return result
                # A valid empty response supersedes any exception from an
                # earlier attempt; terminal classification must reflect the
                # final observable response.
                last_error = None
                last_empty = result
                if attempt < max_attempts:
                    time.sleep(backoff * (2 ** (attempt - 1)))
            except Exception as error:
                last_error = error
                logger.debug(
                    "Chat request failed on attempt {}/{}: {}",
                    attempt,
                    max_attempts,
                    error,
                )
                if attempt < max_attempts:
                    time.sleep(backoff * (2 ** (attempt - 1)))
        if last_empty is not None and last_error is None:
            last_empty.latency_seconds = round(time.time() - started, 3)
            return last_empty
        error_name = type(last_error).__name__ if last_error else "UnknownError"
        return ChatResult(
            api_status=("timeout" if "timeout" in error_name.lower() else "error"),
            attempts=max_attempts,
            error_class=error_name,
            error_message=str(last_error or "Unknown API failure"),
            latency_seconds=round(time.time() - started, 3),
        )


class CommonAnswerHarness:
    """One reader and one prompt shared by every decomposition condition."""

    def __init__(self, config: Mapping[str, Any], client: Any | None = None) -> None:
        self.config = dict(config)
        self.chat = ObservableChatRunner(self.config, client=client)

    def answer(
        self,
        question: str,
        delivered: Sequence[Mapping[str, Any]],
        media_blocks: Sequence[Mapping[str, Any]],
        question_type: str | None = None,
    ) -> ChatResult:
        try:
            messages = build_answer_messages(
                question, delivered, media_blocks, question_type=question_type,
            )
        except Exception as error:
            return ChatResult(
                api_status="error",
                error_class=type(error).__name__,
                error_message=str(error),
            )
        return self.chat.complete(messages)
