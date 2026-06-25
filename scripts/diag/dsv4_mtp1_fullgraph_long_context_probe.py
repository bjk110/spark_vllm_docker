#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# DeepSeek-V4-Flash  MTP(n=1) + FULL_DECODE_ONLY  LONG-CONTEXT probe / driver.
#
# Phase-gated, single-invocation client for the staged long-context campaign
# (LC0 16K -> LC1 32K -> LC2 64K -> LC3 128K/131K). It is a CLIENT ONLY: it does NOT
# load the model, start/stop containers, reboot, clear caches, change presets, retry
# failed requests, delete artifacts, or auto-advance to the next stage/phase. Each
# phase requires an explicit, separate invocation. Stop on first violation.
#
# Phases (run individually, in order, never automatically chained):
#   --phase r1check     : verify graph[2] + MTP activation + NET/IB from a logfile the
#                         caller passes (--startup-log); no requests issued.
#   --phase correctness : UTF-8 / Korean / Unicode / deterministic short-answer +
#                         rejection-heavy + a long-context needle-retrieval check.
#   --phase depth       : two fixed depths (--depths d1,d2), tg128, record TTFT +
#                         prefill throughput + API decode t/s + MTP accounting.
#   --phase sentinel    : a single short-128 + single depth-128 streamed pass for drift.
#
# Stage only sets the documented expectation labels (max-model-len / KV) used in the
# summary; it does NOT change any runtime knob (the preset does that).
#
# Outputs (in --out): requests.jsonl, sentinels.jsonl, r1_activation.json,
# stage_summary.json, heartbeat.txt.

import argparse, json, os, re, time, urllib.request, urllib.error

REPL = "�"
MAX_REPL_FRAC = 0.02
MAX_RUN = 24
NUM_SPEC = 1
M_DR = "vllm:spec_decode_num_drafts"
M_DT = "vllm:spec_decode_num_draft_tokens"
M_AC = "vllm:spec_decode_num_accepted_tokens"
MEM_STOP_GIB = 12.0
MEM_WARN_GIB = 16.0
DISK_STOP_GIB = 20.0

STAGE_SPEC = {  # documentation labels only (the preset enforces the real values).
    # "depths" are INPUT-DEPTH targets (max input tokens). max_model_len is the CORRECTED
    # ceiling that covers max-depth + 128 output + margin (block-256 aligned); it is NOT
    # equal to the named depth. The authoritative effective KV size is taken from
    # --kv-bytes (the preset value) or the runtime-reported capacity, NOT from kv_gib:
    # kv_gib here is only a planning hint and may differ from the run's actual KV.
    "lc0":  {"max_model_len": 18432,  "kv_gib": 2, "depths": [8192, 16384]},
    "lc1":  {"max_model_len": 36864,  "kv_gib": 4, "depths": [16384, 32768]},
    "lc2":  {"max_model_len": 69632,  "kv_gib": 4, "depths": [32768, 65536]},  # 4 GiB (was 8)
    "lc3":  {"max_model_len": 135168, "kv_gib": 4, "depths": [65536, 131072]},  # 4 GiB (was 16)
    # prefill-optimization campaign: identical MML/KV to lc3; the ONLY runtime difference is
    # the preset's MAX_NUM_BATCHED_TOKENS (2048 baseline vs 4096/8192 variants). Screening
    # depths 32K/64K then 131K confirmation. kv_gib is a planning hint only (real KV = 4 GiB).
    "prefill": {"max_model_len": 135168, "kv_gib": 4, "depths": [32768, 65536, 131072]},
}

# deterministic context filler (~1 token / short word for this text)
_FILL = ("In distributed long-context inference the key value cache grows with sequence "
         "length while the decode step reuses cached keys and values for every prior "
         "position before emitting the next token in the running sequence. ")
NEEDLE_FACT = "The secret verification code for this passage is ZEBRA-7741."
# Strong response contract: identifier only, first line, no code, no reasoning, no prose.
# The previous 32-token / weak-contract prompt let the model wander into off-task code on the
# 64K Korean probe; this contract plus a 128-token budget constrains the output format.
NEEDLE_QUESTION = ("\n\nQuestion: What is the secret verification code stored in the passage above?\n"
                   "Return only the exact identifier stored in the document. Put it on the first "
                   "output line. Do not explain, do not output code, do not add any other text.\nAnswer:")
# Korean/Unicode retrieval variant (fake code only); the answer token ZEBRA-7741 stays ASCII
NEEDLE_FACT_KO = "이 문서의 비밀 검증 코드는 ZEBRA-7741 입니다. 한글 유니코드 확인용 문장입니다."
NEEDLE_QUESTION_KO = ("\n\n질문: 위 문서에 저장된 비밀 검증 코드는 무엇입니까?\n"
                      "문서에 저장된 정확한 식별자만 첫 줄에 출력하세요. 설명하지 말고, 코드를 작성하지 말고, "
                      "다른 텍스트를 추가하지 마세요.\nAnswer:")
# Multi-needle (fake codes) placed near 20/50/80 percent; answer must list all three
MULTI_CODES = ["ZEBRA-7741", "FALCON-3920", "MARLIN-8456"]
MULTI_FACTS = [f"Synthetic marker {i+1}: the verification code at this position is {c}."
               for i, c in enumerate(MULTI_CODES)]
MULTI_QUESTION = ("\n\nQuestion: List the three verification codes that appear in the passage.\n"
                  "Return only the three exact identifiers, separated by commas, in the order they "
                  "appear, on the first output line. Do not explain, do not output code, do not add "
                  "any other text.\nAnswer:")
MULTI_FRACS = [0.2, 0.5, 0.8]


import unicodedata as _ud

# Synthetic verification-code identifier shape: e.g. ZEBRA-7741, FALCON-3920, MARLIN-8456.
_ID_RE = re.compile(r"[A-Z]{2,}-\d{4}")


def _norm(s):
    # NFKC-normalize and strip surrounding whitespace for format-tolerant matching.
    return _ud.normalize("NFKC", s or "").strip()


def validate_single(text, expected):
    """Strict but format-tolerant single-needle validation.

    correct == True iff exactly the expected identifier is present and NO unexpected
    synthetic identifier appears. Partial / absent / wrong / extra-synthetic all fail.
    Response-format compliance (first line, extra text, contract) is recorded separately
    and never relaxes correctness."""
    norm = _norm(text)
    ids = _ID_RE.findall(norm)
    expected_present = expected in ids
    unexpected = [i for i in ids if i != expected]
    correct = expected_present and not unexpected
    lines = norm.splitlines()
    first = lines[0] if lines else ""
    on_first_line = expected in _ID_RE.findall(first)
    contract_followed = (_norm(first) == expected)  # first line is exactly the identifier
    info = {"ids_found": ids, "unexpected_ids": unexpected,
            "on_first_line": on_first_line, "extra_text": (norm != expected),
            "contract_followed": contract_followed}
    return correct, info


def validate_multi(text, expected_list):
    """Strict but format-tolerant multi-needle validation.

    correct == True iff all expected identifiers are present and NO unexpected synthetic
    identifier appears. Ordering is recorded as a FORMAT-compliance signal (the prompt asks
    for document order) but does NOT gate correctness, so a correct-but-reordered answer is
    a format warning, not a retrieval failure. Reported separately per protocol."""
    norm = _norm(text)
    ids = _ID_RE.findall(norm)
    all_present = all(e in ids for e in expected_list)
    unexpected = [i for i in ids if i not in expected_list]
    correct = all_present and not unexpected
    # first-occurrence order of expected ids
    seen = []
    for i in ids:
        if i in expected_list and i not in seen:
            seen.append(i)
    order_ok = (seen == list(expected_list))
    lines = norm.splitlines()
    first = lines[0] if lines else ""
    first_ids = _ID_RE.findall(first)
    on_first_line = all(e in first_ids for e in expected_list)
    contract_followed = (_norm(first).replace(" ", "") ==
                         ",".join(expected_list).replace(" ", ""))
    info = {"ids_found": ids, "unexpected_ids": unexpected, "order_ok": order_ok,
            "on_first_line": on_first_line,
            "extra_text": (norm != ", ".join(expected_list)),
            "contract_followed": contract_followed}
    return correct, info


# Verbatim copy of the previous run's failed 64K-Korean completion, kept as a negative
# regression fixture: the corrected validator MUST classify it as not-correct.
_PREV_FAILED_COMPLETION = ('"""\n    )\n    response = requests.post(\n        '
                           'f"http://{host}:{port}/v1/chat/completions",\n        json={\n           ')


def phase_selftest(out):
    """Offline validator unit/static tests. No API call is made (harness calibration)."""
    cases = []

    def chk(name, cond):
        cases.append({"name": name, "pass": bool(cond)})

    c, _ = validate_single("ZEBRA-7741", "ZEBRA-7741"); chk("single_exact_only", c)
    c, _ = validate_single("  ZEBRA-7741  ", "ZEBRA-7741"); chk("single_surrounding_whitespace", c)
    c, i = validate_single("ZEBRA-7741\nextra trailing text here", "ZEBRA-7741")
    chk("single_first_line_then_extra", c and i["on_first_line"] and i["extra_text"])
    c, _ = validate_single("```\nZEBRA-7741\n```", "ZEBRA-7741"); chk("single_in_code_block", c)
    c, _ = validate_single("there is no code here", "ZEBRA-7741"); chk("single_missing_fails", not c)
    c, _ = validate_single("FALCON-3920", "ZEBRA-7741"); chk("single_wrong_id_fails", not c)
    c, _ = validate_single("ZEBRA-774", "ZEBRA-7741"); chk("single_partial_fails", not c)
    c, _ = validate_single("ZEBRA-7741 ZEBRA-9999", "ZEBRA-7741"); chk("single_unexpected_synthetic_fails", not c)
    c, _ = validate_single("정답: ZEBRA-7741 입니다", "ZEBRA-7741"); chk("single_korean_unicode_norm", c)
    c, _ = validate_single(_PREV_FAILED_COMPLETION, "ZEBRA-7741"); chk("single_prev_failed_completion_regression", not c)
    c, _ = validate_multi("ZEBRA-7741, FALCON-3920, MARLIN-8456", MULTI_CODES); chk("multi_exact_ordered", c)
    c, i = validate_multi("FALCON-3920, ZEBRA-7741, MARLIN-8456", MULTI_CODES)
    chk("multi_reordered_correct_format_warn", c and not i["order_ok"])
    c, _ = validate_multi("ZEBRA-7741, FALCON-3920", MULTI_CODES); chk("multi_missing_fails", not c)
    c, _ = validate_multi("ZEBRA-7741, FALCON-3920, MARLIN-8456, BOGUS-0001", MULTI_CODES); chk("multi_unexpected_fails", not c)

    allp = all(x["pass"] for x in cases)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "selftest.json"), "w") as f:
        json.dump({"all_pass": allp, "n": len(cases), "cases": cases}, f,
                  ensure_ascii=False, indent=2)
    print(json.dumps({"all_pass": allp, "n": len(cases),
                      "failed": [x["name"] for x in cases if not x["pass"]]}, ensure_ascii=False))
    return 0 if allp else 2


TOKENIZER = None  # set by --tokenizer; enables exact tokenizer-feedback sizing


def _assemble(reps, with_needle, needle_frac=0.5, lang="en", multi=False):
    body = ("Long-context passage for depth testing. " + (_FILL * reps))
    if multi:
        # insert the 3 fake-code markers near 20/50/80 percent, back-to-front so earlier
        # insertion offsets stay valid
        for frac, fact in sorted(zip(MULTI_FRACS, MULTI_FACTS), reverse=True):
            cut = int(len(body) * frac)
            body = body[:cut] + " " + fact + " " + body[cut:]
        return body + MULTI_QUESTION
    if with_needle:
        fact = NEEDLE_FACT_KO if lang == "ko" else NEEDLE_FACT
        q = NEEDLE_QUESTION_KO if lang == "ko" else NEEDLE_QUESTION
        cut = int(len(body) * needle_frac)
        body = body[:cut] + " " + fact + " " + body[cut:]
        return body + q
    return body + "\n\nBriefly summarize the passage above in one sentence:"


def build_depth_prompt(approx_tokens, with_needle=False, needle_frac=0.5):
    # Fallback empirical sizing (~38 tok/_FILL, ~12 overhead) used ONLY when no exact
    # tokenizer is loaded. Deliberately undershoots; never relies on server truncation.
    reps = max(1, int((approx_tokens - 12 - (15 if with_needle else 0)) / 38))
    return _assemble(reps, with_needle, needle_frac)


def _ntok(reps, with_needle, needle_frac=0.5, lang="en", multi=False):
    # exact prompt-token count from the official tokenizer with API special-token path
    return len(TOKENIZER.encode(_assemble(reps, with_needle, needle_frac, lang, multi)))


def build_exact_prompt(target_tokens, with_needle=False, needle_frac=0.5, tol=64,
                       lang="en", multi=False):
    # Tokenizer-feedback exact sizing: bounded binary search (LOCAL tokenization only,
    # NO API calls) for the largest `reps` whose exact token count <= target_tokens.
    # Token count is monotonic non-decreasing in reps. Returns (prompt, actual_tokens,
    # reps). Raises if it cannot land within `tol` below the target.
    if TOKENIZER is None:
        raise RuntimeError("build_exact_prompt requires --tokenizer")
    # Bounded upper limit: ~38 tokens/_FILL repeat, so the reps needed for `target` is
    # ~target/38. Set hi = target//25 (a safe over-estimate of the needed reps) so the
    # largest candidate probe is ~1.5x the target tokens — NEVER a multi-million-token
    # string that would exceed the model's 1,048,576 context limit and emit a warning.
    lo, hi = 1, max(2, target_tokens // 25)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _ntok(mid, with_needle, needle_frac, lang, multi) <= target_tokens:
            lo = mid
        else:
            hi = mid - 1
    reps = lo
    actual = _ntok(reps, with_needle, needle_frac, lang, multi)
    if actual > target_tokens:
        raise RuntimeError(f"exact-sizing overshoot reps={reps} actual={actual}>{target_tokens}")
    if target_tokens - actual > tol and _ntok(reps + 1, with_needle, needle_frac, lang, multi) <= target_tokens:
        # one more rep would still fit and tighten tolerance
        reps += 1; actual = _ntok(reps, with_needle, needle_frac, lang, multi)
    return _assemble(reps, with_needle, needle_frac, lang, multi), actual, reps


def make_prompt(target_tokens, with_needle=False, needle_frac=0.5, lang="en"):
    # prefer exact tokenizer-feedback sizing; fall back to empirical if no tokenizer
    if TOKENIZER is not None:
        p, _, _ = build_exact_prompt(target_tokens, with_needle, needle_frac, lang=lang)
        return p
    return build_depth_prompt(target_tokens, with_needle, needle_frac)


def post(base, payload, timeout=600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(base + "/v1/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return r.status, raw, (time.perf_counter() - t0) * 1e3


def post_stream(base, payload, timeout=600):
    payload = dict(payload); payload["stream"] = True
    # include_usage emits a trailing chunk (choices=[]) carrying usage.prompt_tokens, the
    # ACTUAL API prompt-token count. We use it for TTFT-derived prefill throughput so the
    # value is the server's count, not the local tokenizer estimate.
    payload["stream_options"] = {"include_usage": True}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(base + "/v1/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); t_first = None; t_last = None; ntok = 0; text = []
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            try:
                obj = json.loads(p)
            except Exception:
                continue
            u = obj.get("usage")
            if u:
                usage = u  # trailing usage chunk (or per-chunk usage); keep the last seen
            ch = obj.get("choices") or []
            tk = ch[0].get("text", "") if ch else ""
            now = time.perf_counter()
            if tk:
                if t_first is None:
                    t_first = now
                t_last = now; ntok += 1; text.append(tk)
    ttft_ms = (t_first - t0) * 1e3 if t_first else None
    dec_s = (t_last - t_first) if (t_first and t_last and t_last > t_first) else None
    dec_tps = ((ntok - 1) / dec_s) if (dec_s and ntok > 1) else None
    return ttft_ms, dec_tps, ntok, "".join(text), usage


def get_health(base):
    try:
        with urllib.request.urlopen(base + "/health", timeout=10) as r:
            return r.status
    except Exception as e:
        return f"ERR:{e}"


def scrape_spec(base):
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
            body = r.read().decode("utf-8", "replace")
    except Exception:
        return None
    out = {}
    for k in (M_DR, M_DT, M_AC):
        tot = None
        for m in re.finditer(r"^" + re.escape(k) + r"_total\{[^}]*\}\s+([0-9eE.+-]+)\s*$",
                             body, re.MULTILINE):
            tot = (tot or 0.0) + float(m.group(1))
        out[k] = tot
    return out


def read_mem():
    mi = open("/proc/meminfo").read()
    def g(n):
        m = re.search(r"^%s:\s+(\d+)" % n, mi, re.MULTILINE)
        return int(m.group(1)) / 1048576 if m else None
    vm = open("/proc/vmstat").read()
    def v(n):
        m = re.search(r"^%s (\d+)" % n, vm, re.MULTILINE)
        return int(m.group(1)) if m else None
    return {"MemAvailable": g("MemAvailable"),
            "SwapUsed": (g("SwapTotal") or 0) - (g("SwapFree") or 0),
            "pswpin": v("pswpin"), "pswpout": v("pswpout"),
            "pgmajfault": v("pgmajfault")}


def disk_free_gib(path="/home"):
    try:
        s = os.statvfs(path)
        return s.f_bavail * s.f_frsize / (1024**3)
    except Exception:
        return None


def check_integrity(raw):
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        return False, f"invalid_utf8:{e}", None, {}
    try:
        d = json.loads(text)
    except Exception as e:
        return False, f"non_json:{e}", None, {}
    ch = d.get("choices", [{}])[0]; out = ch.get("text", ""); fin = ch.get("finish_reason")
    usage = d.get("usage", {}); meta = {"finish_reason": fin, "usage": usage}
    if not out:
        return False, "empty_output", out, meta
    if out.count(REPL) > max(1, int(len(out) * MAX_REPL_FRAC)):
        return False, "replacement_flood", out, meta
    run = mx = 1
    for i in range(1, len(out)):
        run = run + 1 if out[i] == out[i-1] else 1
        mx = max(mx, run)
    if mx > MAX_RUN:
        return False, f"repetition_{mx}", out, meta
    if (usage.get("completion_tokens") or 0) == 0:
        return False, "zero_completion", out, meta
    if fin not in ("length", "stop"):
        return False, f"bad_finish:{fin}", out, meta
    return True, "ok", out, meta


def check_acct(before, after, ctok):
    if not before or not after or any(before.get(k) is None or after.get(k) is None
                                      for k in (M_DR, M_DT, M_AC)):
        return False, "metrics_unavailable", {}
    d = after[M_DR]-before[M_DR]; dt = after[M_DT]-before[M_DT]; ac = after[M_AC]-before[M_AC]
    deltas = {"drafts": d, "draft_tokens": dt, "accepted": ac,
              "accept_rate": (ac/dt) if dt else None}
    if d < 0 or dt < 0 or ac < 0:
        return False, "counter_decreased", deltas
    if dt != d * NUM_SPEC:
        return False, f"dt_ne_drafts:{dt}!={d}", deltas
    if not (0 <= ac <= dt):
        return False, f"accepted_oob:{ac}/{dt}", deltas
    return True, "ok", deltas


def mem_gate():
    """Return (fatal, reason) per the stop policy. Never auto-recovers."""
    m = read_mem(); df = disk_free_gib("/home")
    if m["MemAvailable"] is not None and m["MemAvailable"] < MEM_STOP_GIB:
        return True, f"MemAvailable<{MEM_STOP_GIB} ({m['MemAvailable']:.1f})"
    if df is not None and df < DISK_STOP_GIB:
        return True, f"disk_free<{DISK_STOP_GIB} ({df:.1f})"
    return False, None


def run_req(base, out, seq, tag, prompt, max_tokens, expect_needle=None, stream=False):
    hb = get_health(base); mb = scrape_spec(base)
    try:
        if stream:
            ttft, dtps, ntok, txt, usage = post_stream(base, {"model": MODEL, "prompt": prompt,
                "max_tokens": max_tokens, "temperature": 0, "top_p": 1.0, "seed": 0})
            st, raw, ms, ok, reason, meta = 200, b"", None, (ntok > 0), ("ok" if ntok else "empty"), \
                ({"usage": usage} if usage else {})
        else:
            st, raw, ms = post(base, {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                                      "temperature": 0, "top_p": 1.0, "seed": 0, "stream": False})
            ok, reason, txt, meta = check_integrity(raw)
            ttft = dtps = ntok = None
    except urllib.error.HTTPError as e:
        # an HTTP error (e.g. 400 context-limit) is a recorded request failure, NOT a crash
        st, raw, ms = e.code, b"", None
        ok, reason, txt, meta = False, f"http_error_{e.code}", None, {}
        ttft = dtps = ntok = None
    ma = scrape_spec(base); ha = get_health(base); m = read_mem()
    aok, areason, deltas = check_acct(mb, ma, None)
    needle_ok = None
    if expect_needle is not None and txt is not None:
        needle_ok = expect_needle in txt
    api_ptok = (meta.get("usage", {}) or {}).get("prompt_tokens")
    # TTFT-derived prefill throughput: actual API prompt tokens / TTFT seconds.
    # Labeled explicitly as TTFT-derived; this is NOT a native server prompt-processing rate.
    ttft_prefill_tps = (round(api_ptok / (ttft / 1e3), 1)
                        if (api_ptok and ttft and ttft > 0) else None)
    rec = {"seq": seq, "tag": tag, "ts": time.strftime("%H:%M:%S"), "http": st,
           "ms": round(ms, 1) if ms else None, "ttft_ms": round(ttft, 1) if ttft else None,
           "decode_tps": round(dtps, 2) if dtps else None, "ntok": ntok,
           "ttft_prefill_tps": ttft_prefill_tps,
           "ok": ok, "reason": reason, "acct_ok": aok, "acct_reason": areason,
           "drafts": deltas.get("drafts"), "draft_tokens": deltas.get("draft_tokens"),
           "accepted": deltas.get("accepted"), "acceptance": deltas.get("accept_rate"),
           "prompt_tokens": api_ptok,
           "completion_tokens": (meta.get("usage", {}) or {}).get("completion_tokens"),
           "needle_ok": needle_ok, "health_before": hb, "health_after": ha,
           "mem_after": m["MemAvailable"], "swap_after": m["SwapUsed"],
           "text": (txt or "")[:120]}
    with open(os.path.join(out, "requests.jsonl"), "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps({k: rec[k] for k in ("seq", "tag", "http", "ok", "acct_ok",
          "ttft_ms", "decode_tps", "needle_ok", "mem_after")}, ensure_ascii=False), flush=True)
    fatal = (not ok) or (not aok) or st != 200 or ha != 200 or (needle_ok is False)
    # Return the FULL completion text to the caller (disk record stays truncated to 120
    # chars). Retrieval validation needs the full body, not the preview.
    rec["text"] = txt or ""
    return fatal, rec


def phase_r1check(out, startup_log):
    """Verify graph[2] + MTP + NET/IB from a caller-provided startup logfile."""
    if not startup_log or not os.path.exists(startup_log):
        print("ERROR: --startup-log required and must exist for r1check"); return 2
    log = open(startup_log, "r", errors="replace").read()
    proof = {
        "ts": time.strftime("%H:%M:%S"),
        "full_decode_only": "FULL_DECODE_ONLY" in log,
        "capture_size_2": "cudagraph_capture_sizes': [2]" in log or "cudagraph_capture_sizes\":[2]" in log,
        "graph_captured": bool(re.search(r"Graph capturing finished", log)),
        "mtp_loaded": "MTP draft model loaded" in log,
        "next_n_2": "next_n=2" in log,
        "netib": ("NET/IB : Using" in log) and ("Using network IB" in log),
        "no_socket_fallback": "NET/Socket" not in log,
        "no_fallback": "falling back to eager" not in log,
        "no_recapture": not re.search(r"recaptur", log, re.I),
    }
    proof["r1_pass"] = all(v is True for k, v in proof.items() if k != "ts")
    with open(os.path.join(out, "r1_activation.json"), "w") as f:
        json.dump(proof, f, ensure_ascii=False, indent=2)
    print(json.dumps(proof, ensure_ascii=False))
    return 0 if proof["r1_pass"] else 2


def phase_correctness(base, out, depth):
    seq = 0; rej = 0
    basics = [("arith", "2+2=", 16), ("english", "The capital of France is", 16),
              ("korean", "대한민국의 수도는", 16),
              ("unicode", "Emoji test: \U0001f600 and 漢字 and Ω — continue:", 16),
              ("hex", "Echo exactly, character by character: 9af3c7e1d05b8246af19", 32),
              ("shuffled", "Output exactly: lorem zxqw ipsum vbnm dolor plok sit qwer amet", 32)]
    for tag, p, mx in basics:
        seq += 1
        fatal, rec = run_req(base, out, seq, "corr_" + tag, p, mx)
        if tag in ("hex", "shuffled") and (rec.get("draft_tokens") or 0) > (rec.get("accepted") or 0):
            rej += 1
        if fatal:
            return 2, f"correctness_{tag}_{rec['reason']}/{rec['acct_reason']}"
        fg, fr = mem_gate()
        if fg:
            return 2, fr
    # long-context needle/retrieval at this stage depth
    seq += 1
    npmt = make_prompt(depth, with_needle=True)
    fatal, rec = run_req(base, out, seq, f"needle_d{depth}", npmt, 32, expect_needle="ZEBRA-7741")
    if fatal:
        return 2, f"needle_fail seq={seq} needle_ok={rec.get('needle_ok')}"
    if rej < 1:
        return 2, "no_rejection_observed"
    return 0, "ok"


def phase_depth(base, out, depths, max_model_len, kv_cap):
    # token-exact, non-streamed first to capture usage (prompt/completion tokens), then a
    # streamed pass for TTFT/decode t/s. No API truncation option is used. A request whose
    # measured total would meet/exceed MML or KV capacity is a fatal arithmetic failure.
    res = []
    for d in depths:
        fg, fr = mem_gate()
        if fg:
            return 2, fr, res
        seq = len(res) + 1
        if TOKENIZER is not None:
            prompt, local_actual, reps = build_exact_prompt(d)
        else:
            prompt, local_actual, reps = make_prompt(d), None, None
        # measured (non-stream) for exact token accounting
        f1, r1 = run_req(base, out, seq, f"depth_meas_d{d}", prompt, 128, stream=False)
        ptok = r1.get("prompt_tokens"); ctok = r1.get("completion_tokens")
        total = (ptok or 0) + (ctok or 0)
        margin_mml = (max_model_len - total) if ptok is not None else None
        margin_kv = (kv_cap - total) if (kv_cap and ptok is not None) else None
        # streamed for TTFT + decode t/s
        f2, r2 = run_req(base, out, seq, f"depth_perf_d{d}", prompt, 128, stream=True)
        res.append({"depth": d, "local_actual_tokens": local_actual, "reps": reps,
                    "local_vs_target_diff": (local_actual - d) if local_actual is not None else None,
                    "prompt_tokens": ptok, "completion_tokens": ctok,
                    "total_tokens": total, "max_model_len": max_model_len,
                    "margin_to_mml": margin_mml, "kv_cap": kv_cap,
                    "margin_to_kv_cap": margin_kv,
                    "ttft_ms": r2["ttft_ms"], "decode_tps": r2["decode_tps"], "ntok": r2["ntok"]})
        # arithmetic-failure guards: no truncation, no boundary
        if ptok is not None and total >= max_model_len:
            return 2, f"context_limit total={total}>=MML={max_model_len} d={d}", res
        if kv_cap and ptok is not None and total >= kv_cap:
            return 2, f"context_limit total={total}>=KVcap={kv_cap} d={d}", res
        if f1 or f2:
            return 2, f"depth_{d}_{r1['reason']}/{r2['reason']}", res
    return 0, "ok", res


def _stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"mean": None, "std": None, "cv": None, "n": 0}
    n = len(xs); mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    std = var ** 0.5
    return {"mean": round(mean, 3), "std": round(std, 3),
            "cv": round(std / mean, 4) if mean else None, "n": n}


def phase_retrieval(base, out, depth, langs=("en", "ko"), multi=True):
    # Deterministic synthetic-needle retrieval at `depth`. The probe set is explicit:
    # `langs` selects single-needle languages and `multi` adds the 20/50/80-percent case.
    # Every retrieval probe uses a 128-token output budget. Validation is strict but
    # format-tolerant (validate_single / validate_multi); retrieval correctness and response
    # format compliance are recorded SEPARATELY (format never relaxes correctness).
    res = []
    cases = [(l, f"retr_{l}", False) for l in langs]
    if multi:
        cases.append(("en", "retr_multi", True))
    for lang, tag, is_multi in cases:
        fg, fr = mem_gate()
        if fg:
            return 2, fr, res
        if TOKENIZER is not None:
            prompt, local_actual, reps = build_exact_prompt(depth, with_needle=not is_multi,
                                                            lang=lang, multi=is_multi)
        else:
            prompt, local_actual = make_prompt(depth, with_needle=not is_multi), None
        fatal, rec = run_req(base, out, len(res) + 1, f"{tag}_d{depth}", prompt, 128)
        text = rec.get("text", "") or ""
        if is_multi:
            correct, info = validate_multi(text, MULTI_CODES)
            res.append({"case": "multi", "depth": depth, "local_actual": local_actual,
                        "api_prompt_tokens": rec.get("prompt_tokens"),
                        "codes_found": info["ids_found"], "all_found": correct,
                        "unexpected_ids": info["unexpected_ids"],
                        "fmt_on_first_line": info["on_first_line"],
                        "fmt_order_ok": info["order_ok"],
                        "fmt_extra_text": info["extra_text"],
                        "fmt_contract_followed": info["contract_followed"],
                        "http": rec.get("http"), "text_preview": text[:160]})
            if fatal or not correct:
                return 2, f"retrieval_multi_fail found={info['ids_found']} unexpected={info['unexpected_ids']}", res
        else:
            correct, info = validate_single(text, "ZEBRA-7741")
            res.append({"lang": lang, "depth": depth, "local_actual": local_actual,
                        "api_prompt_tokens": rec.get("prompt_tokens"), "needle_ok": correct,
                        "ids_found": info["ids_found"], "unexpected_ids": info["unexpected_ids"],
                        "fmt_on_first_line": info["on_first_line"],
                        "fmt_extra_text": info["extra_text"],
                        "fmt_contract_followed": info["contract_followed"],
                        "http": rec.get("http"), "text_preview": text[:160]})
            if fatal or not correct:
                return 2, f"retrieval_{lang}_fail needle_ok={correct} ids={info['ids_found']}", res
    return 0, "ok", res


def phase_retrieval_matrix(base, out):
    # Protocol-conformant cold-reproduction retrieval matrix (corrected harness):
    #   1) 64K English single needle
    #   2) 131K English single needle
    #   3) 131K Korean/Unicode single needle
    #   4) 131K multi-position 20/50/80-percent needles
    # Deliberately NO 64K Korean probe; the previous 64K-ko failure stays preserved as
    # historical evidence in the prior (inconclusive) result directory.
    rc, reason, a = phase_retrieval(base, out, 65536, langs=("en",), multi=False)
    if rc:
        return rc, reason, a
    rc, reason, b = phase_retrieval(base, out, 131072, langs=("en", "ko"), multi=True)
    return rc, reason, a + b


def phase_perf(base, out, depths, runs=3):
    # one unmeasured warm-up + `runs` measured streamed requests per depth (tg128),
    # concurrency 1, identical prompt construction. Reports mean/std/CV.
    allres = []
    for d in depths:
        fg, fr = mem_gate()
        if fg:
            return 2, fr, allres
        prompt = make_prompt(d)
        # warm-up (excluded)
        post_stream(base, {"model": MODEL, "prompt": prompt, "max_tokens": 128,
                           "temperature": 0, "top_p": 1.0, "seed": 0})
        ttfts = []; dtps = []; pftps = []; ptoks = []; samples = []
        for r in range(runs):
            seq = len(allres) * 100 + r + 1
            fatal, rec = run_req(base, out, seq, f"perf_d{d}_r{r}", prompt, 128, stream=True)
            ttfts.append(rec["ttft_ms"]); dtps.append(rec["decode_tps"])
            if rec.get("ttft_prefill_tps") is not None:
                pftps.append(rec["ttft_prefill_tps"])
            if rec.get("prompt_tokens") is not None:
                ptoks.append(rec["prompt_tokens"])
            samples.append({"ttft_ms": rec["ttft_ms"], "decode_tps": rec["decode_tps"],
                            "ttft_prefill_tps": rec.get("ttft_prefill_tps"),
                            "api_prompt_tokens": rec.get("prompt_tokens"), "ntok": rec["ntok"]})
            if fatal:
                return 2, f"perf_{d}_r{r}_{rec['reason']}", allres
        # TTFT-derived prefill throughput = actual API prompt_tokens / TTFT seconds (per run).
        allres.append({"depth": d, "runs": runs, "samples": samples,
                       "api_prompt_tokens": (ptoks[-1] if ptoks else None),
                       "ttft_ms": _stats(ttfts), "decode_tps": _stats(dtps),
                       "ttft_prefill_tps": _stats(pftps)})
    return 0, "ok", allres


def phase_repeat(base, out, depth, count=5):
    # bounded immediate-repetition drift check at a fixed depth (no soak beyond `count`).
    res = []
    for i in range(count):
        fg, fr = mem_gate()
        if fg:
            return 2, fr, res
        fatal, rec = run_req(base, out, i + 1, f"repeat_d{depth}_{i}",
                             make_prompt(depth), 128, stream=True)
        res.append({"i": i, "ttft_ms": rec["ttft_ms"], "decode_tps": rec["decode_tps"],
                    "ttft_prefill_tps": rec.get("ttft_prefill_tps"),
                    "api_prompt_tokens": rec.get("prompt_tokens"),
                    "ntok": rec["ntok"], "mem_after": rec["mem_after"],
                    "swap_after": rec["swap_after"]})
        if fatal:
            return 2, f"repeat_{i}_{rec['reason']}", res
    return 0, "ok", res


def phase_sentinel(base, out):
    recs = []
    for tag, prompt in (("short", "Count slowly: one, two, three,"),
                        ("depth4096", make_prompt(4096))):
        ttft, dtps, ntok, _ = post_stream(base, {"model": MODEL, "prompt": prompt,
            "max_tokens": 128, "temperature": 0, "top_p": 1.0, "seed": 0})
        recs.append({"tag": tag, "ttft_ms": round(ttft, 1) if ttft else None,
                     "decode_tps": round(dtps, 2) if dtps else None, "ntok": ntok})
    with open(os.path.join(out, "sentinels.jsonl"), "a") as f:
        f.write(json.dumps({"ts": time.strftime("%H:%M:%S"), "sentinels": recs},
                           ensure_ascii=False) + "\n")
    print(json.dumps(recs, ensure_ascii=False))
    return 0, "ok"


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=list(STAGE_SPEC.keys()), required=True)
    ap.add_argument("--phase", choices=["r1check", "correctness", "depth", "perf",
                                        "repeat", "retrieval", "retrieval_matrix",
                                        "selftest", "sentinel"], required=True)
    ap.add_argument("--repeat-depth", type=int, default=16384)
    ap.add_argument("--repeat-count", type=int, default=5)
    ap.add_argument("--retr-depth", type=int, default=0)
    ap.add_argument("--startup-log", default="")
    ap.add_argument("--depths", default="",
                    help="comma-separated token depths; defaults to the stage's depths")
    ap.add_argument("--kv-cap", type=int, default=0,
                    help="reported KV token capacity (from startup log) for margin checks")
    ap.add_argument("--kv-bytes", type=int, default=0,
                    help="effective fixed KV bytes from the active preset (authoritative)")
    ap.add_argument("--tokenizer", default="",
                    help="path to the official tokenizer for exact prompt sizing")
    args = ap.parse_args(); MODEL = args.model
    global TOKENIZER
    if args.tokenizer:
        from transformers import AutoTokenizer
        TOKENIZER = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    os.makedirs(args.out, exist_ok=True)
    spec = STAGE_SPEC[args.stage]
    depths = [int(x) for x in args.depths.split(",")] if args.depths else spec["depths"]

    rc = 0; reason = "ok"; extra = {}
    if args.phase == "r1check":
        rc = phase_r1check(args.out, args.startup_log)
        reason = "r1_pass" if rc == 0 else "r1_fail"
    elif args.phase == "correctness":
        rc, reason = phase_correctness(args.base, args.out, spec["depths"][-1])
    elif args.phase == "depth":
        rc, reason, extra = phase_depth(args.base, args.out, depths,
                                        spec["max_model_len"], args.kv_cap)
    elif args.phase == "perf":
        rc, reason, extra = phase_perf(args.base, args.out, depths)
    elif args.phase == "repeat":
        rc, reason, extra = phase_repeat(args.base, args.out, args.repeat_depth,
                                         args.repeat_count)
    elif args.phase == "retrieval":
        rd = args.retr_depth or spec["depths"][-1]
        rc, reason, extra = phase_retrieval(args.base, args.out, rd)
    elif args.phase == "retrieval_matrix":
        rc, reason, extra = phase_retrieval_matrix(args.base, args.out)
    elif args.phase == "selftest":
        rc = phase_selftest(args.out)
        reason = "selftest_pass" if rc == 0 else "selftest_fail"
    elif args.phase == "sentinel":
        rc, reason = phase_sentinel(args.base, args.out)

    m = read_mem()
    summary = {"stage": args.stage, "phase": args.phase,
               "max_model_len": spec["max_model_len"],
               "effective_kv_bytes": args.kv_bytes or None,
               "effective_kv_gib": round(args.kv_bytes / 1073741824, 2) if args.kv_bytes else None,
               "kv_capacity_tokens": args.kv_cap or None,
               "kv_gib_planning_hint": spec["kv_gib"],
               "depths": depths, "rc": rc, "reason": reason,
               "MemAvailable": round(m["MemAvailable"], 1) if m["MemAvailable"] else None,
               "SwapUsed": round(m["SwapUsed"], 2), "disk_free_gib": round(disk_free_gib() or 0, 1),
               "extra": extra}
    with open(os.path.join(args.out, "stage_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "heartbeat.txt"), "w") as f:
        f.write(json.dumps({"DONE": True, "stage": args.stage, "phase": args.phase,
                            "rc": rc, "reason": reason}, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False))
    # NOTE: never auto-advances to the next phase/stage.
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
