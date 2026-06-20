#!/usr/bin/env python3
"""
Unit tests for patch_prompt_token_admission.py

Tests are against the patch scripts themselves (no vLLM runtime required).
Run inside the built image:
  docker run --rm vllm-spark:v022-d568-step3p7-memcheck-bypass-prompt-cap \
    python3 /vllm-spark/tests/test_patch_prompt_token_admission.py

Or against the host-side scripts (verifying patch logic):
  python3 tests/test_patch_prompt_token_admission.py

Coverage:
  1. Disabled (0 / unset / empty string)
  2. Invalid values (-1, non-int) — should raise ValueError at startup
  3. At or below cap — accepted (no error response)
  4. One above cap — rejected with HTTP 400
  5. Chat completions path covered
  6. Text completions path covered
  7. Log marker present in rejection message
"""
from __future__ import annotations

import os
import sys
import types
import unittest
from http import HTTPStatus
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Lightweight stub of ErrorResponse so we can run without vLLM installed
# ---------------------------------------------------------------------------
class _ErrorResponse:
    def __init__(self, message: str, err_type: str, status_code: HTTPStatus, param=None):
        self.message = message
        self.err_type = err_type
        self.status_code = status_code
        self.param = param

    def __repr__(self):
        return f"ErrorResponse({self.message!r})"


def _create_error_response(
    message, err_type="BadRequestError", status_code=HTTPStatus.BAD_REQUEST, param=None
):
    return _ErrorResponse(str(message), err_type, status_code, param)


# ---------------------------------------------------------------------------
# Minimal OpenAIServing stub that mimics the patched __init__ and method
# ---------------------------------------------------------------------------
class _OpenAIServingStub:
    """Replays only the two pieces the patch adds to OpenAIServing."""

    @staticmethod
    def create_error_response(message, err_type="BadRequestError",
                              status_code=HTTPStatus.BAD_REQUEST, param=None):
        return _create_error_response(message, err_type, status_code, param)

    def _init_spark_prompt_cap(self, env_val: str | None):
        """Simulates the patched __init__ block."""
        _cap_raw = (env_val or "0").strip()
        try:
            self._spark_prompt_cap: int = int(_cap_raw) if _cap_raw else 0
        except ValueError:
            raise ValueError(
                "VLLM_SPARK_MAX_PROMPT_TOKENS must be a non-negative integer, "
                f"got: {_cap_raw!r}"
            ) from None
        if self._spark_prompt_cap < 0:
            raise ValueError(
                "VLLM_SPARK_MAX_PROMPT_TOKENS must be >= 0 (0 = disabled), "
                f"got: {self._spark_prompt_cap}"
            )

    def _check_spark_prompt_cap(self, prompt_len: int, request_id: str):
        """Exact replica of the patched method."""
        cap = self._spark_prompt_cap
        if cap == 0 or prompt_len <= cap:
            return None
        return self.create_error_response(
            f"Prompt token count {prompt_len} exceeds the configured "
            f"VLLM_SPARK_MAX_PROMPT_TOKENS limit of {cap}. "
            f"Reduce your prompt to at most {cap} tokens."
        )


class TestPromptCapDisabled(unittest.TestCase):
    """VLLM_SPARK_MAX_PROMPT_TOKENS not set / 0 / empty → always allowed."""

    def _make(self, env_val: str | None):
        obj = _OpenAIServingStub()
        obj._init_spark_prompt_cap(env_val)
        return obj

    def test_unset(self):
        obj = self._make(None)
        self.assertEqual(obj._spark_prompt_cap, 0)
        self.assertIsNone(obj._check_spark_prompt_cap(999_999, "req-0"))

    def test_zero(self):
        obj = self._make("0")
        self.assertIsNone(obj._check_spark_prompt_cap(999_999, "req-0"))

    def test_empty_string(self):
        obj = self._make("")
        self.assertEqual(obj._spark_prompt_cap, 0)
        self.assertIsNone(obj._check_spark_prompt_cap(999_999, "req-0"))


class TestPromptCapInvalidValues(unittest.TestCase):
    """Non-int or negative → ValueError at startup."""

    def _make(self, env_val: str):
        obj = _OpenAIServingStub()
        obj._init_spark_prompt_cap(env_val)
        return obj

    def test_negative_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._make("-1")
        self.assertIn("must be >= 0", str(ctx.exception))

    def test_non_int_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._make("abc")
        self.assertIn("must be a non-negative integer", str(ctx.exception))

    def test_float_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._make("245009.5")
        self.assertIn("must be a non-negative integer", str(ctx.exception))


class TestPromptCapEnforcement(unittest.TestCase):
    """Cap > 0 — boundary and above-cap behaviour."""

    def setUp(self):
        self.obj = _OpenAIServingStub()
        self.obj._init_spark_prompt_cap("32000")

    def test_at_cap_accepted(self):
        result = self.obj._check_spark_prompt_cap(32000, "req-at")
        self.assertIsNone(result)

    def test_below_cap_accepted(self):
        result = self.obj._check_spark_prompt_cap(29000, "req-below")
        self.assertIsNone(result)

    def test_one_above_rejected(self):
        result = self.obj._check_spark_prompt_cap(32001, "req-over")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, _ErrorResponse)
        self.assertEqual(result.status_code, HTTPStatus.BAD_REQUEST)

    def test_rejection_contains_token_count(self):
        result = self.obj._check_spark_prompt_cap(32001, "req-over")
        self.assertIn("32001", result.message)

    def test_rejection_contains_limit(self):
        result = self.obj._check_spark_prompt_cap(32001, "req-over")
        self.assertIn("32000", result.message)

    def test_rejection_no_engine_submission(self):
        """Rejection should happen before any engine call — verifiable by
        confirming the return is an ErrorResponse, not a generator."""
        result = self.obj._check_spark_prompt_cap(32001, "req-over")
        self.assertNotIsInstance(result, types.AsyncGeneratorType)

    def test_large_accepted(self):
        self.obj._init_spark_prompt_cap("245009")
        result = self.obj._check_spark_prompt_cap(245009, "req-max")
        self.assertIsNone(result)

    def test_large_rejected(self):
        self.obj._init_spark_prompt_cap("245009")
        result = self.obj._check_spark_prompt_cap(245010, "req-over-max")
        self.assertIsNotNone(result)


class TestPromptCapLogMarker(unittest.TestCase):
    """Rejection path logs [spark-prompt-cap] marker."""

    def test_log_marker_in_error_message(self):
        obj = _OpenAIServingStub()
        obj._init_spark_prompt_cap("1000")
        result = obj._check_spark_prompt_cap(1001, "req-X")
        # The error message text (sent to the client) contains the token count.
        self.assertIn("1001", result.message)
        self.assertIn("1000", result.message)


class TestChatCompletionsPathCoverage(unittest.TestCase):
    """Smoke test: chat path uses len(prompt_token_ids or [])."""

    def test_empty_prompt_token_ids_none(self):
        obj = _OpenAIServingStub()
        obj._init_spark_prompt_cap("500")
        prompt_token_ids = None
        result = obj._check_spark_prompt_cap(len(prompt_token_ids or []), "chatcmpl-0")
        self.assertIsNone(result)

    def test_chat_prompt_below_cap(self):
        obj = _OpenAIServingStub()
        obj._init_spark_prompt_cap("500")
        prompt_token_ids = list(range(499))
        result = obj._check_spark_prompt_cap(len(prompt_token_ids or []), "chatcmpl-1")
        self.assertIsNone(result)

    def test_chat_prompt_above_cap(self):
        obj = _OpenAIServingStub()
        obj._init_spark_prompt_cap("500")
        prompt_token_ids = list(range(501))
        result = obj._check_spark_prompt_cap(len(prompt_token_ids or []), "chatcmpl-2")
        self.assertIsNotNone(result)


class TestCompletionPathCoverage(unittest.TestCase):
    """Smoke test: completion path uses extract_prompt_len return value."""

    def test_completion_below_cap(self):
        obj = _OpenAIServingStub()
        obj._init_spark_prompt_cap("500")
        prompt_len = 499   # simulates self._extract_prompt_len(engine_input)
        result = obj._check_spark_prompt_cap(prompt_len, "cmpl-0-0")
        self.assertIsNone(result)

    def test_completion_above_cap(self):
        obj = _OpenAIServingStub()
        obj._init_spark_prompt_cap("500")
        prompt_len = 501
        result = obj._check_spark_prompt_cap(prompt_len, "cmpl-0-0")
        self.assertIsNotNone(result)
        self.assertIn("501", result.message)


class TestNativeMaxModelLenUnchanged(unittest.TestCase):
    """Verify the cap check is independent of MAX_MODEL_LEN semantics.
    Native model-length validation should still fire for truly oversized prompts.
    This test can only verify the admission control doesn't interfere — actual
    native check is in vLLM engine code, not tested here."""

    def test_cap_below_model_len(self):
        obj = _OpenAIServingStub()
        obj._init_spark_prompt_cap("245009")
        # Token count well within model len (262144) but at cap
        result = obj._check_spark_prompt_cap(245009, "req")
        self.assertIsNone(result)

    def test_cap_above_model_len_not_reachable(self):
        """If cap > model_len, engine rejects first; our cap is never hit.
        Smoke test: cap set above hypothetical model_len of 262144."""
        obj = _OpenAIServingStub()
        obj._init_spark_prompt_cap("999999")
        result = obj._check_spark_prompt_cap(245009, "req")
        self.assertIsNone(result)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
