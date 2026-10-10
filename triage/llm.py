"""Thin OpenAI-compatible chat client.

Kept behind a tiny interface (`complete(messages) -> str`) so tests can swap in a
fake and so the pipeline never depends on SDK specifics.
"""

from __future__ import annotations

import logging
import threading
from typing import Protocol

from openai import BadRequestError, OpenAI

from .config import Settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Transport-level failure (timeout, connection refused, 5xx...)."""


class ChatModel(Protocol):
    calls: int

    def complete(self, messages: list[dict]) -> str: ...


class OpenAICompatibleModel:
    def __init__(self, settings: Settings):
        self.settings = settings
        # SDK retries disabled: retry policy lives in the pipeline, where we can
        # also feed validation errors back to the model.
        self._client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout_s,
            max_retries=0,
        )
        self._lock = threading.Lock()
        self.calls = 0  # requests sent
        self.successes = 0  # requests that returned a completion
        self._json_mode = True  # dropped automatically if the server rejects it

    def complete(self, messages: list[dict]) -> str:
        kwargs = dict(
            model=self.settings.llm_model,
            messages=messages,
            temperature=self.settings.llm_temperature,
            max_tokens=700,
        )
        if self._json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        with self._lock:
            self.calls += 1
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            if self._json_mode and "response_format" in str(e).lower():
                log.warning("Server rejected response_format=json_object; continuing without it.")
                self._json_mode = False
                return self.complete(messages)
            raise LLMError(str(e)) from e
        except Exception as e:  # network, timeout, 5xx - all treated as transient
            raise LLMError(f"{type(e).__name__}: {e}") from e
        with self._lock:
            self.successes += 1
        choice = resp.choices[0] if resp.choices else None
        return (choice.message.content or "") if choice and choice.message else ""
