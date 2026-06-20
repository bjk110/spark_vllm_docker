#!/usr/bin/env python3
"""
Add VLLM_SPARK_MAX_PROMPT_TOKENS admission control to the vLLM 0.22 OpenAI
serving layer.

Why this patch exists
---------------------
The Step-3.7-Flash-NVFP4 v0.22 EP-on/mp/CUDA-graph path on dual GB10 Spark
nodes has a validated prompt-token ceiling of 245,009 tokens (Stage D, 2026-06-20).
A 257,891-token prompt reproducibly caused an infrastructure hang (node unresponsive,
reboot required). MAX_MODEL_LEN=262,144 is the engine limit, not the validated
operational ceiling.

This patch adds a server-side gate (VLLM_SPARK_MAX_PROMPT_TOKENS) that rejects
requests exceeding the configured limit with HTTP 400 before they reach the
engine — covering both /v1/chat/completions and /v1/completions.

How it works
------------
1. engine/serving.py (OpenAIServing.__init__):
   - Reads VLLM_SPARK_MAX_PROMPT_TOKENS at startup (int, default 0 = disabled).
   - Validates: non-negative int; raises ValueError on startup if invalid.
   - Logs INFO if cap > 0 so operators know admission control is active.
   - Stores as self._spark_prompt_cap.

2. engine/serving.py (new method _check_spark_prompt_cap):
   - Returns None if disabled (cap == 0) or prompt_len <= cap.
   - Returns HTTP 400 ErrorResponse with actual count + limit if exceeded.
   - Logs WARNING with request_id, prompt_tokens, limit on rejection.

3. chat_completion/serving.py (_create_chat_completion):
   - Calls _check_spark_prompt_cap after prompt_token_ids are extracted,
     before engine_client.generate. Returns the ErrorResponse immediately.

4. completion/serving.py (_create_completion):
   - Calls _check_spark_prompt_cap in the engine_inputs loop,
     before engine_client.generate. Returns the ErrorResponse immediately.

Patch is idempotent — MARKER presence short-circuits re-runs at each site.
All three patches are atomic: all anchors are verified before any write.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Target paths
# ---------------------------------------------------------------------------
ENGINE_SERVING = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/engine/serving.py"
)
CHAT_SERVING = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py"
)
COMPLETION_SERVING = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/completion/serving.py"
)

MARKER = "_spark_prompt_cap"

# ---------------------------------------------------------------------------
# Patch 1: OpenAIServing.__init__ — read env var, validate, store as instance var
# ---------------------------------------------------------------------------
ENGINE_INIT_OLD = """\
        except Exception:
            # Never fail server startup over the fingerprint.
            self.system_fingerprint = None

    @staticmethod
    def create_error_response("""

ENGINE_INIT_NEW = """\
        except Exception:
            # Never fail server startup over the fingerprint.
            self.system_fingerprint = None

        # vllm-spark patch: VLLM_SPARK_MAX_PROMPT_TOKENS prompt-token admission control.
        # Read once at startup; 0 = disabled (default). Raises ValueError on bad value.
        import os as _os
        _cap_raw = _os.environ.get("VLLM_SPARK_MAX_PROMPT_TOKENS", "0").strip()
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
        if self._spark_prompt_cap > 0:
            from vllm.logger import init_logger as _init_logger
            _init_logger(__name__).info(
                "[spark-prompt-cap] admission control active: "
                "max_prompt_tokens=%d",
                self._spark_prompt_cap,
            )

    @staticmethod
    def create_error_response("""

# ---------------------------------------------------------------------------
# Patch 2: new method _check_spark_prompt_cap — inserted before _raise_if_error
# ---------------------------------------------------------------------------
ENGINE_METHOD_OLD = """\
        return json_str

    def _raise_if_error(self, finish_reason: str | None, request_id: str) -> None:"""

ENGINE_METHOD_NEW = """\
        return json_str

    def _check_spark_prompt_cap(
        self, prompt_len: int, request_id: str
    ) -> "ErrorResponse | None":
        \"\"\"Return HTTP 400 ErrorResponse if prompt exceeds VLLM_SPARK_MAX_PROMPT_TOKENS.\"\"\"
        cap = self._spark_prompt_cap
        if cap == 0 or prompt_len <= cap:
            return None
        from vllm.logger import init_logger as _init_logger
        _init_logger(__name__).warning(
            "[spark-prompt-cap] rejected request %s: prompt_tokens=%d limit=%d",
            request_id,
            prompt_len,
            cap,
        )
        return self.create_error_response(
            f"Prompt token count {prompt_len} exceeds the configured "
            f"VLLM_SPARK_MAX_PROMPT_TOKENS limit of {cap}. "
            f"Reduce your prompt to at most {cap} tokens."
        )

    def _raise_if_error(self, finish_reason: str | None, request_id: str) -> None:"""

# ---------------------------------------------------------------------------
# Patch 3: chat_completion/serving.py — check after sub_request_id, before max_tokens
# ---------------------------------------------------------------------------
CHAT_OLD = """\
            sub_request_id = (
                request_id if len(engine_inputs) == 1 else f"{request_id}_{i}"
            )

            max_tokens = get_max_tokens(
                max_model_len,
                request.max_completion_tokens"""

CHAT_NEW = """\
            sub_request_id = (
                request_id if len(engine_inputs) == 1 else f"{request_id}_{i}"
            )

            # vllm-spark patch: prompt-token admission control
            _cap_err = self._check_spark_prompt_cap(
                len(prompt_token_ids or []), sub_request_id
            )
            if _cap_err is not None:
                return _cap_err

            max_tokens = get_max_tokens(
                max_model_len,
                request.max_completion_tokens"""

# ---------------------------------------------------------------------------
# Patch 4: completion/serving.py — check at top of engine_inputs loop
# ---------------------------------------------------------------------------
COMPLETION_OLD = """\
        for i, engine_input in enumerate(engine_inputs):
            max_tokens = get_max_tokens(
                max_model_len,
                request.max_tokens,
                self._extract_prompt_len(engine_input),"""

COMPLETION_NEW = """\
        for i, engine_input in enumerate(engine_inputs):
            # vllm-spark patch: prompt-token admission control
            _cap_err = self._check_spark_prompt_cap(
                self._extract_prompt_len(engine_input), f"{request_id}-{i}"
            )
            if _cap_err is not None:
                return _cap_err

            max_tokens = get_max_tokens(
                max_model_len,
                request.max_tokens,
                self._extract_prompt_len(engine_input),"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read(p: Path) -> str:
    if not p.exists():
        raise FileNotFoundError(f"Target not found: {p}")
    return p.read_text()


def verify_anchor(src: str, anchor: str, label: str) -> None:
    if anchor not in src:
        raise RuntimeError(
            f"[patch_prompt_token_admission] Anchor not found in {label}.\n"
            "vLLM version mismatch — update patch anchors.\n"
            f"Anchor starts with: {anchor[:80]!r}"
        )


def apply_patch(src: str, old: str, new: str) -> str:
    return src.replace(old, new, 1)


def validate_syntax(src: str, path: Path) -> None:
    try:
        ast.parse(src)
    except SyntaxError as exc:
        raise RuntimeError(
            f"[patch_prompt_token_admission] Syntax error after patching {path}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    engine_src = read(ENGINE_SERVING)
    chat_src = read(CHAT_SERVING)
    completion_src = read(COMPLETION_SERVING)

    # Idempotency: all three markers must be absent for a fresh apply.
    already = [
        MARKER in engine_src,
        MARKER in chat_src,
        MARKER in completion_src,
    ]
    if all(already):
        print(
            "[patch_prompt_token_admission] all three sites already patched — no-op"
        )
        return 0
    if any(already):
        sites = ["engine/serving.py", "chat_completion/serving.py", "completion/serving.py"]
        partial = [s for s, a in zip(sites, already) if a]
        missing = [s for s, a in zip(sites, already) if not a]
        print(
            f"[patch_prompt_token_admission] PARTIAL PATCH DETECTED — "
            f"marker present in {partial} but absent in {missing}. "
            "Manual inspection required. Aborting."
        )
        return 1

    # Verify all anchors BEFORE writing anything.
    verify_anchor(engine_src, ENGINE_INIT_OLD, "engine/serving.py (init)")
    verify_anchor(engine_src, ENGINE_METHOD_OLD, "engine/serving.py (method)")
    verify_anchor(chat_src, CHAT_OLD, "chat_completion/serving.py")
    verify_anchor(completion_src, COMPLETION_OLD, "completion/serving.py")
    print("[patch_prompt_token_admission] all anchors verified")

    # Apply patches.
    engine_patched = apply_patch(engine_src, ENGINE_INIT_OLD, ENGINE_INIT_NEW)
    engine_patched = apply_patch(engine_patched, ENGINE_METHOD_OLD, ENGINE_METHOD_NEW)
    chat_patched = apply_patch(chat_src, CHAT_OLD, CHAT_NEW)
    completion_patched = apply_patch(completion_src, COMPLETION_OLD, COMPLETION_NEW)

    # Validate syntax of all three.
    validate_syntax(engine_patched, ENGINE_SERVING)
    validate_syntax(chat_patched, CHAT_SERVING)
    validate_syntax(completion_patched, COMPLETION_SERVING)
    print("[patch_prompt_token_admission] syntax validated")

    # Write all three.
    ENGINE_SERVING.write_text(engine_patched)
    CHAT_SERVING.write_text(chat_patched)
    COMPLETION_SERVING.write_text(completion_patched)

    print(
        "[patch_prompt_token_admission] applied:\n"
        f"  {ENGINE_SERVING}\n"
        f"  {CHAT_SERVING}\n"
        f"  {COMPLETION_SERVING}\n"
        "VLLM_SPARK_MAX_PROMPT_TOKENS admission control is now active "
        "(set to >0 to enable; default 0 = disabled)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
