#!/usr/bin/env python3
"""
Unit tests for the correctness checker verdict logic used in
bench-bt-matrix-step37-v023.sh correctness_check_extended().

Run: python3 benchmarks/tests/test_correctness_checker.py
  or: python3 -m pytest benchmarks/tests/test_correctness_checker.py -v
"""

import re
import sys


# ---------------------------------------------------------------------------
# Mirror of garble detection embedded in _classify() (test_type == garble)
# Kept in sync with the Python code inside the bash here-string.
# ---------------------------------------------------------------------------

def _pua_range():
    return re.compile("[" + chr(0xe000) + "-" + chr(0xf8ff) + "]")

_PUA_RE = _pua_range()
_GARBLE_RE = re.compile("[" + chr(0xfffd) + chr(0xd800) + "-" + chr(0xdfff) + "]",
                         re.UNICODE)


def is_garble(text: str) -> bool:
    """Return True if the text contains garble signals.

    Signals (in precedence order):
      1. U+FFFD replacement character or lone surrogates (U+D800-U+DFFF)
      2. Private Use Area (U+E000-U+F8FF) exceeding 10% of text length
      3. Pathological repetition: same 3-15 char sequence repeated 5+ times

    Characters outside ASCII/Korean range — e.g. U+00B7 MIDDLE DOT —
    are NOT garble signals.
    """
    try:
        if _GARBLE_RE.search(text):
            return True
    except Exception:
        return True  # unparseable text treated as garble

    pua = len(_PUA_RE.findall(text))
    if pua > max(5, len(text) * 0.1):
        return True

    for n in range(3, 16):
        pattern = "(.{" + str(n) + r"})\1{4}"
        if re.search(pattern, text):
            return True

    return False


# ---------------------------------------------------------------------------
# Verdict classification
# ---------------------------------------------------------------------------

def classify_verdict(content: str, finish_reason: str,
                     expected_grep: str, test_type: str) -> str:
    """Classify one API response.

    Verdict taxonomy:
      PASS                    - answer found; finish_reason=stop
      PASS_WITH_LENGTH_LIMIT  - answer found; finish_reason=length (response truncated)
      FAIL_WRONG_ANSWER       - output present but expected pattern absent; stop
      FAIL_GARBLE             - corruption signal (garble test type only)
      INCONCLUSIVE_OUTPUT_BUDGET - finish_reason=length; pattern not confirmed
      FAIL_API                - parse error, HTTP error, or empty stop response
    """
    if "PARSE_ERROR" in content or finish_reason == "error":
        return "FAIL_API"
    if not content and finish_reason == "stop":
        return "FAIL_API"
    if not content:
        return "INCONCLUSIVE_OUTPUT_BUDGET"
    if test_type == "garble":
        if is_garble(content):
            return "FAIL_GARBLE"
        return "INCONCLUSIVE_OUTPUT_BUDGET" if finish_reason == "length" else "PASS"
    if finish_reason == "length":
        return ("PASS_WITH_LENGTH_LIMIT"
                if re.search(expected_grep, content)
                else "INCONCLUSIVE_OUTPUT_BUDGET")
    if re.search(expected_grep, content):
        return "PASS"
    return "FAIL_WRONG_ANSWER"


# ---------------------------------------------------------------------------
# Per-test best aggregation
# ---------------------------------------------------------------------------

_PRECEDENCE = {
    "PASS": 6,
    "PASS_WITH_LENGTH_LIMIT": 5,
    "FAIL_GARBLE": 4,
    "FAIL_WRONG_ANSWER": 3,
    "FAIL_API": 2,
    "INCONCLUSIVE_OUTPUT_BUDGET": 1,
}


def best_verdict(verdicts: list) -> str:
    """Reduce per-run verdicts to the best overall verdict for a test."""
    best = "INCONCLUSIVE_OUTPUT_BUDGET"
    for v in verdicts:
        if _PRECEDENCE.get(v, 0) > _PRECEDENCE.get(best, 0):
            best = v
        if best == "PASS":
            break
    return best


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

KOREAN_KTX_RESPONSE = (
    "서울에서 부산까지 KTX의 "
    "소요시간은 출발·도착 "
    "역과 정차역, 운행 열차 "
    "종류에 따라 조금 차이가 "
    "있습니다."
)
# Decoded: "서울에서 부산까지 KTX의 소요시간은 출발·도착 역과 정차역, 운행 열차 종류에 따라 조금 차이가 있습니다."
# Note: · = U+00B7 MIDDLE DOT — legitimate Korean punctuation, must NOT trigger garble.

VALID_KOREAN = "서울에서 부산까지 KTX 소요시간은 약 2시간 30분입니다."
REPLACEMENT_CHAR_TEXT = "정상텍스트�가나다"
# chr(0xd800) is a lone surrogate — use with PYTHONLEGACYWINDOWSSTDIO or catch
VALID_ANSWER_97 = "The answer is 97."
VALID_ANSWER_FACTORIAL = "15! = 1307674368000"
WRONG_ANSWER = "The answer is 42."
# PUA-heavy text: > 10% PUA chars
PUA_HEAVY_TEXT = "".join(chr(0xe000 + i) for i in range(20)) + "abc"
# Pathological repetition: "가나다" × 10 (30 chars of 3-char pattern repeated 10 times)
REPEAT_TEXT = "가나다" * 10


# ---------------------------------------------------------------------------
# Tests — is_garble()
# ---------------------------------------------------------------------------

def test_no_garble_plain_korean():
    assert not is_garble(VALID_KOREAN)

def test_no_garble_middle_dot():
    """U+00B7 MIDDLE DOT in Korean text must NOT trigger garble."""
    assert not is_garble(KOREAN_KTX_RESPONSE), (
        "Korean text with U+00B7 (middle dot) was incorrectly classified as garble"
    )

def test_no_garble_ascii():
    assert not is_garble("The answer is 97.")

def test_no_garble_mixed():
    assert not is_garble("KTX 소요시간은 2h30m")

def test_garble_replacement_char():
    assert is_garble(REPLACEMENT_CHAR_TEXT)

def test_garble_pua_dominance():
    assert is_garble(PUA_HEAVY_TEXT)

def test_garble_pua_small_count_ok():
    """A single PUA char in long text should not trigger garble."""
    text = VALID_KOREAN * 10 + chr(0xe001)
    assert not is_garble(text)

def test_garble_pathological_repetition():
    assert is_garble(REPEAT_TEXT)

def test_no_garble_normal_repetition():
    """A word repeated 3 times (well below the 5-repetition threshold)."""
    text = "서울 서울 서울"
    assert not is_garble(text)

def test_no_garble_latin_extended():
    """Latin Extended chars like U+00C9 (É) should not trigger garble."""
    assert not is_garble("café au lait")

def test_no_garble_empty():
    assert not is_garble("")


# ---------------------------------------------------------------------------
# Tests — classify_verdict()
# ---------------------------------------------------------------------------

def test_case_1_pass_stop_correct():
    """Case 1: correct answer + stop -> PASS"""
    assert classify_verdict(VALID_ANSWER_97, "stop", "97", "exact") == "PASS"

def test_case_2_inconclusive_output_budget():
    """Case 2: valid output + length + answer NOT found -> INCONCLUSIVE_OUTPUT_BUDGET"""
    assert classify_verdict(VALID_KOREAN, "length", "97", "exact") == "INCONCLUSIVE_OUTPUT_BUDGET"

def test_case_3_pass_with_length_limit():
    """Case 3: correct answer found + length -> PASS_WITH_LENGTH_LIMIT"""
    assert classify_verdict(VALID_ANSWER_97, "length", "97", "exact") == "PASS_WITH_LENGTH_LIMIT"

def test_case_4_fail_wrong_answer():
    """Case 4: valid output + stop + wrong answer -> FAIL_WRONG_ANSWER"""
    assert classify_verdict(WRONG_ANSWER, "stop", "97", "exact") == "FAIL_WRONG_ANSWER"

def test_case_5_fail_garble_replacement():
    """Case 5: replacement char -> FAIL_GARBLE"""
    assert classify_verdict(REPLACEMENT_CHAR_TEXT, "stop", ".", "garble") == "FAIL_GARBLE"

def test_case_6_fail_garble_repetition():
    """Case 6: pathological repetition -> FAIL_GARBLE"""
    assert classify_verdict(REPEAT_TEXT, "stop", ".", "garble") == "FAIL_GARBLE"

def test_case_7_fail_api_parse_error():
    """Case 7: parse error -> FAIL_API"""
    assert classify_verdict("PARSE_ERROR:connection refused", "error", ".", "exact") == "FAIL_API"

def test_case_8_empty_length():
    """Case 8: empty content + length (budget exhausted) -> INCONCLUSIVE_OUTPUT_BUDGET"""
    assert classify_verdict("", "length", ".", "exact") == "INCONCLUSIVE_OUTPUT_BUDGET"

def test_case_9_empty_stop():
    """Case 9: empty content + stop -> FAIL_API"""
    assert classify_verdict("", "stop", ".", "exact") == "FAIL_API"

def test_bt8192_test3_run1():
    """Reproduce bt=8192 Test 3 Run 1.
    Old checker: FAIL_GARBLE (false positive from U+00B7 middle dot).
    New checker: INCONCLUSIVE_OUTPUT_BUDGET (finish_reason=length, no garble).
    """
    v = classify_verdict(KOREAN_KTX_RESPONSE, "length", ".", "garble")
    assert v == "INCONCLUSIVE_OUTPUT_BUDGET", (
        f"bt=8192 Test3 Run1: expected INCONCLUSIVE_OUTPUT_BUDGET, got {v!r}"
    )

def test_garble_test_stop_valid_korean():
    """Garble test + valid Korean + stop -> PASS"""
    assert classify_verdict(VALID_KOREAN, "stop", ".", "garble") == "PASS"

def test_garble_test_pua_heavy():
    """PUA-heavy text is garble regardless of finish_reason."""
    assert classify_verdict(PUA_HEAVY_TEXT, "stop", ".", "garble") == "FAIL_GARBLE"
    assert classify_verdict(PUA_HEAVY_TEXT, "length", ".", "garble") == "FAIL_GARBLE"


# ---------------------------------------------------------------------------
# Tests — best_verdict()
# ---------------------------------------------------------------------------

def test_best_pass_dominates():
    assert best_verdict(["INCONCLUSIVE_OUTPUT_BUDGET", "PASS"]) == "PASS"
    assert best_verdict(["FAIL_GARBLE", "PASS"]) == "PASS"
    assert best_verdict(["FAIL_WRONG_ANSWER", "PASS"]) == "PASS"

def test_best_pass_with_length_limit_over_fail():
    assert best_verdict(["INCONCLUSIVE_OUTPUT_BUDGET", "PASS_WITH_LENGTH_LIMIT"]) == "PASS_WITH_LENGTH_LIMIT"
    assert best_verdict(["FAIL_WRONG_ANSWER", "PASS_WITH_LENGTH_LIMIT"]) == "PASS_WITH_LENGTH_LIMIT"
    assert best_verdict(["FAIL_GARBLE", "PASS_WITH_LENGTH_LIMIT"]) == "PASS_WITH_LENGTH_LIMIT"

def test_best_fail_garble():
    assert best_verdict(["FAIL_GARBLE", "INCONCLUSIVE_OUTPUT_BUDGET"]) == "FAIL_GARBLE"
    assert best_verdict(["FAIL_GARBLE", "FAIL_WRONG_ANSWER"]) == "FAIL_GARBLE"

def test_best_inconclusive():
    assert best_verdict(["INCONCLUSIVE_OUTPUT_BUDGET", "INCONCLUSIVE_OUTPUT_BUDGET"]) == "INCONCLUSIVE_OUTPUT_BUDGET"

def test_best_bt8192_test3():
    """bt=8192 Test 3: Run 1=INCONCLUSIVE_OUTPUT_BUDGET (revised), Run 2=PASS -> PASS"""
    assert best_verdict(["INCONCLUSIVE_OUTPUT_BUDGET", "PASS"]) == "PASS"

def test_best_single_pass():
    assert best_verdict(["PASS"]) == "PASS"

def test_best_single_fail_garble():
    assert best_verdict(["FAIL_GARBLE"]) == "FAIL_GARBLE"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_tests():
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for name, func in tests:
        try:
            func()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
