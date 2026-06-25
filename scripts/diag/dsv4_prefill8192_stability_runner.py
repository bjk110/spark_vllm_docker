#!/usr/bin/env python3
"""DeepSeek-V4 prefill8192 candidate — 60-minute mixed-context stability runner.

This is a SEPARATE orchestration harness. It imports the frozen long-context probe driver
read-only (for deterministic prompt construction, /v1/completions streaming, spec-decode
metric scraping, accounting, memory sampling, and the retrieval validator) and does NOT
modify it; the driver file SHA stays unchanged. It runs a fixed-order mixed workload at
concurrency 1 for a bounded wall-clock window, stops on the first failure with no retry, and
writes per-request JSONL plus a summary.

Fixed-order cycle:
  1) one 32K performance request   (128 output tokens, streamed)
  2) one 64K performance request   (128 output tokens, streamed)
  3) one 131K performance request  (128 output tokens, streamed)
  4) one short correctness sentinel
  5) every fifth cycle: one 131K English retrieval request (strict-but-tolerant validation)

No calibration request. No retry. No adaptive pacing beyond a fixed inter-request pause.
"""
import argparse
import importlib.util
import json
import os
import time


def load_driver(path):
    spec = importlib.util.spec_from_file_location("drv", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def perf_request(drv, base, out_dir, cycle, tag, prompt, jsonl):
    """One streamed performance/sentinel request. Returns (ok, rec). ok=False is a hard stop."""
    hb = drv.get_health(base)
    mb = drv.scrape_spec(base)
    ttft, dtps, ntok, text, usage = drv.post_stream(base, {
        "model": drv.MODEL, "prompt": prompt, "max_tokens": 128,
        "temperature": 0, "top_p": 1.0, "seed": 0})
    ma = drv.scrape_spec(base)
    ha = drv.get_health(base)
    m = drv.read_mem()
    aok, areason, deltas = drv.check_acct(mb, ma, None)
    api_ptok = (usage or {}).get("prompt_tokens")
    pf = round(api_ptok / (ttft / 1e3), 1) if (api_ptok and ttft) else None
    rec = {"cycle": cycle, "tag": tag, "ts": time.strftime("%H:%M:%S"),
           "http": 200 if ntok and ntok > 0 else 0, "ntok": ntok,
           "ttft_ms": round(ttft, 1) if ttft else None,
           "ttft_prefill_tps": pf, "decode_tps": round(dtps, 2) if dtps else None,
           "api_prompt_tokens": api_ptok,
           "drafts": deltas.get("drafts"), "accepted": deltas.get("accepted"),
           "acceptance": deltas.get("accept_rate"),
           "acct_ok": aok, "acct_reason": areason,
           "health_before": hb, "health_after": ha,
           "MemAvailable": round(m["MemAvailable"], 2) if m["MemAvailable"] else None,
           "SwapUsed_gib": round(m["SwapUsed"], 2), "pswpin": m["pswpin"],
           "pswpout": m["pswpout"], "pgmajfault": m["pgmajfault"],
           "disk_free_gib": round(drv.disk_free_gib() or 0, 1)}
    jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n"); jsonl.flush()
    ok = (ntok and ntok > 0) and aok and ha == 200
    return ok, rec


def retrieval_request(drv, base, cycle, prompt, jsonl):
    """One streamed 131K English retrieval request with strict-but-tolerant validation."""
    hb = drv.get_health(base)
    ttft, dtps, ntok, text, usage = drv.post_stream(base, {
        "model": drv.MODEL, "prompt": prompt, "max_tokens": 128,
        "temperature": 0, "top_p": 1.0, "seed": 0})
    ha = drv.get_health(base)
    m = drv.read_mem()
    correct, info = drv.validate_single(text or "", "ZEBRA-7741")
    api_ptok = (usage or {}).get("prompt_tokens")
    rec = {"cycle": cycle, "tag": "stability_retr_en_d131072", "ts": time.strftime("%H:%M:%S"),
           "http": 200 if ntok and ntok > 0 else 0, "ntok": ntok,
           "ttft_ms": round(ttft, 1) if ttft else None,
           "api_prompt_tokens": api_ptok, "needle_ok": correct,
           "ids_found": info["ids_found"], "unexpected_ids": info["unexpected_ids"],
           "fmt_on_first_line": info["on_first_line"], "fmt_extra_text": info["extra_text"],
           "fmt_contract_followed": info["contract_followed"],
           "health_before": hb, "health_after": ha,
           "MemAvailable": round(m["MemAvailable"], 2) if m["MemAvailable"] else None,
           "text_preview": (text or "")[:120]}
    jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n"); jsonl.flush()
    ok = (ntok and ntok > 0) and (ha == 200) and correct
    return ok, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://10.10.10.1:8000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--driver", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--pause", type=float, default=2.0, help="fixed inter-request pause seconds")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    drv = load_driver(args.driver)
    drv.MODEL = "deepseek-ai/DeepSeek-V4-Flash"
    from transformers import AutoTokenizer
    drv.TOKENIZER = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Deterministic prompts built ONCE (identical every cycle).
    p32 = drv.make_prompt(32768)
    p64 = drv.make_prompt(65536)
    p131 = drv.make_prompt(131072)
    sentinel = "Count slowly: one, two, three,"
    retr131, _, _ = drv.build_exact_prompt(131072, with_needle=True, lang="en")

    jsonl = open(os.path.join(args.out, "stability_requests.jsonl"), "a")
    t0 = time.time()
    deadline = t0 + args.minutes * 60
    cycle = 0
    n_ok = 0
    n_fail = 0
    n_retr_ok = 0
    n_retr = 0
    stop_reason = "completed_full_duration"
    steps = [("32K", "stability_perf_d32768", p32),
             ("64K", "stability_perf_d65536", p64),
             ("131K", "stability_perf_d131072", p131),
             ("sentinel", "stability_sentinel", sentinel)]

    while time.time() < deadline:
        cycle += 1
        # hard memory stop-gate before each cycle
        fg, fr = drv.mem_gate()
        if fg:
            stop_reason = f"mem_gate:{fr}"; break
        for _name, tag, prompt in steps:
            ok, rec = perf_request(drv, args.base, args.out, cycle, tag, prompt, jsonl)
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                stop_reason = f"fail:{tag}:cycle{cycle}:acct={rec.get('acct_ok')}:http={rec.get('http')}"
                break
            time.sleep(args.pause)
        if n_fail:
            break
        if cycle % 5 == 0:
            n_retr += 1
            ok, rec = retrieval_request(drv, args.base, cycle, retr131, jsonl)
            if ok:
                n_retr_ok += 1
                n_ok += 1
            else:
                n_fail += 1
                stop_reason = f"fail:retrieval:cycle{cycle}:needle_ok={rec.get('needle_ok')}"
                break
            time.sleep(args.pause)

    jsonl.close()
    elapsed = round(time.time() - t0, 1)
    summary = {"minutes_target": args.minutes, "elapsed_seconds": elapsed,
               "cycles_completed": cycle if n_fail == 0 else cycle - 1,
               "cycles_attempted": cycle,
               "requests_ok": n_ok, "requests_failed": n_fail,
               "retrieval_attempted": n_retr, "retrieval_ok": n_retr_ok,
               "stop_reason": stop_reason}
    with open(os.path.join(args.out, "stability_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
