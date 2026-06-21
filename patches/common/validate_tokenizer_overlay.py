#!/usr/bin/env python3
"""
Validation for the Step-3.7 tokenizer overlay generator
(gen_tokenizer_overlay.py).

Modes
-----
--unit (default): offline guard/unit tests using SYNTHETIC fixtures. No GPU, no
  model weights, no model mount. Exercises every guard in generate(): correct
  source, wrong source hash, wrong class, missing required file, malformed JSON,
  idempotency, stale/partial output replacement, source-untouched, exactly one
  changed field, deterministic output, and path restrictions.

--behavior <overlay_dir>: load the real overlay with AutoTokenizer and verify
  Korean / mixed / whitespace / chat-template / special-token behavior matches
  the validated patched baseline (Korean preserved, round-trip OK, no Ġ/Ċ
  artifacts). Requires Transformers (present in the image); no GPU/weights.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_tokenizer_overlay as G  # noqa: E402

VECTORS = {
    "korean": "대한민국의 수도는 서울입니다.",
    "mixed": "안녕, 세계 / Hello, world",
    "ws_nl": "line1\n\nline2   three  spaces\ttab",
    "punct_kr": "한국어, 그리고!  (괄호) — 끝.",
}


# --------------------------------------------------------------------------- #
# Unit tests (synthetic fixtures)
# --------------------------------------------------------------------------- #
def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _make_fixture(root, orig_class="FakeOrigFast"):
    """Create a synthetic upstream model dir + matching Spec."""
    model = os.path.join(root, "model")
    os.makedirs(model, exist_ok=True)
    cfg = {"tokenizer_class": orig_class, "bos_token": "<s>", "model_max_length": 8192}
    open(os.path.join(model, "tokenizer_config.json"), "w").write(
        json.dumps(cfg, ensure_ascii=False, indent=2))
    verbatim_content = {
        "tokenizer.json": '{"version":"1.0","model":{"type":"BPE"}}',
        "special_tokens_map.json": '{"bos_token":"<s>"}',
        "chat_template.jinja": "{{ messages }}",
    }
    verbatim = {}
    for fn, content in verbatim_content.items():
        open(os.path.join(model, fn), "w").write(content)
        verbatim[fn] = _sha(os.path.join(model, fn))
    # compute expected post-image hash for the transform
    cfg2 = dict(cfg)
    cfg2["tokenizer_class"] = "PreTrainedTokenizerFast"
    out_sha = hashlib.sha256(json.dumps(cfg2, ensure_ascii=False, indent=2).encode()).hexdigest()
    spec = G.Spec(
        src_tokcfg_sha=_sha(os.path.join(model, "tokenizer_config.json")),
        out_tokcfg_sha=out_sha,
        expected_orig_class=orig_class,
        target_class="PreTrainedTokenizerFast",
        verbatim=verbatim,
        model_revision="synthetic",
        allowed_model_roots=(model,),
        allowed_out_roots=(root + os.sep,),
    )
    return model, spec


def _expect_raise(fn, label, results):
    try:
        fn()
        results.append((label, "FAIL (no exception)"))
    except G.OverlayError:
        results.append((label, "PASS"))
    except Exception as e:
        results.append((label, f"FAIL (wrong exc {type(e).__name__}: {e})"))


def unit() -> int:
    results = []
    root = tempfile.mkdtemp(prefix="tokunit.")
    try:
        model, spec = _make_fixture(root)
        out = os.path.join(root, "overlay")
        src_cfg = os.path.join(model, "tokenizer_config.json")
        src_sha_before = _sha(src_cfg)

        # 1. positive
        m = G.generate(model, out, spec)
        ok = (_sha(os.path.join(out, "tokenizer_config.json")) == spec.out_tokcfg_sha
              and m["overlay_tokenizer_class"] == "PreTrainedTokenizerFast")
        results.append(("1_positive", "PASS" if ok else "FAIL"))

        # 2. exactly one field changed
        a = json.load(open(src_cfg)); b = json.load(open(os.path.join(out, "tokenizer_config.json")))
        diffk = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        results.append(("2_one_field_changed", "PASS" if diffk == ["tokenizer_class"] else f"FAIL {diffk}"))

        # 3. source untouched
        results.append(("3_source_untouched", "PASS" if _sha(src_cfg) == src_sha_before else "FAIL"))

        # 4. idempotent (reuse)
        m2 = G.generate(model, out, spec)
        results.append(("4_idempotent", "PASS" if m2["out_tokenizer_config_sha256"] == spec.out_tokcfg_sha else "FAIL"))

        # 5. deterministic (fresh dir, same out hash)
        out2 = os.path.join(root, "overlay2")
        G.generate(model, out2, spec)
        results.append(("5_deterministic",
                        "PASS" if _sha(os.path.join(out2, "tokenizer_config.json")) == _sha(os.path.join(out, "tokenizer_config.json")) else "FAIL"))

        # 6. stale/partial output replaced
        out3 = os.path.join(root, "overlay3")
        os.makedirs(out3); open(os.path.join(out3, "junk"), "w").write("stale")
        open(os.path.join(out3, "overlay_manifest.json"), "w").write('{"out_tokenizer_config_sha256":"deadbeef"}')
        G.generate(model, out3, spec)
        results.append(("6_stale_replaced",
                        "PASS" if (_sha(os.path.join(out3, "tokenizer_config.json")) == spec.out_tokcfg_sha
                                   and not os.path.exists(os.path.join(out3, "junk"))) else "FAIL"))

        # 7. wrong source hash
        badroot = tempfile.mkdtemp(prefix="tokunit-badhash.")
        bmodel, bspec = _make_fixture(badroot)
        with open(os.path.join(bmodel, "tokenizer_config.json"), "a") as f:
            f.write(" ")  # tamper after spec computed
        _expect_raise(lambda: G.generate(bmodel, os.path.join(badroot, "o"), bspec), "7_wrong_src_hash", results)

        # 8. wrong class
        croot = tempfile.mkdtemp(prefix="tokunit-class.")
        cmodel, cspec = _make_fixture(croot, orig_class="SomethingElse")
        cspec2 = G.Spec(**{**cspec.__dict__, "expected_orig_class": "DifferentClass"})
        _expect_raise(lambda: G.generate(cmodel, os.path.join(croot, "o"), cspec2), "8_wrong_class", results)

        # 9. missing required file
        mroot = tempfile.mkdtemp(prefix="tokunit-missing.")
        mmodel, mspec = _make_fixture(mroot)
        os.remove(os.path.join(mmodel, "tokenizer.json"))
        _expect_raise(lambda: G.generate(mmodel, os.path.join(mroot, "o"), mspec), "9_missing_file", results)

        # 10. malformed JSON
        jroot = tempfile.mkdtemp(prefix="tokunit-json.")
        jmodel, jspec = _make_fixture(jroot)
        open(os.path.join(jmodel, "tokenizer_config.json"), "w").write("{not json")
        # recompute spec src hash to match the malformed file so we reach the JSON parse guard
        jspec2 = G.Spec(**{**jspec.__dict__, "src_tokcfg_sha": _sha(os.path.join(jmodel, "tokenizer_config.json"))})
        _expect_raise(lambda: G.generate(jmodel, os.path.join(jroot, "o"), jspec2), "10_malformed_json", results)

        # 11. path restriction: overlay inside model root
        _expect_raise(lambda: G.generate(model, os.path.join(model, "inside"), spec), "11_out_inside_model", results)

        # 12. path restriction: model outside allowed roots
        prspec = G.Spec(**{**spec.__dict__, "allowed_model_roots": ("/nonexistent-root",)})
        _expect_raise(lambda: G.generate(model, out, prspec), "12_model_outside_root", results)

        # 13. unexpected existing output dir (no manifest -> not owned by overlay)
        out_foreign = os.path.join(root, "foreign")
        os.makedirs(out_foreign)
        open(os.path.join(out_foreign, "someone-elses-file"), "w").write("keep me")
        _expect_raise(lambda: G.generate(model, out_foreign, spec), "13_unexpected_existing_dir", results)
        # and confirm the foreign content was not clobbered
        results.append(("13b_foreign_untouched",
                        "PASS" if os.path.exists(os.path.join(out_foreign, "someone-elses-file")) else "FAIL"))

        for br in (badroot, croot, mroot, jroot):
            shutil.rmtree(br, ignore_errors=True)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("[validate_tokenizer_overlay] UNIT:")
    for name, r in results:
        print(f"  {name}: {r}")
    failed = [n for n, r in results if not r.startswith("PASS")]
    print("UNIT_PASS" if not failed else f"UNIT_FAIL: {failed}")
    return 0 if not failed else 1


# --------------------------------------------------------------------------- #
# Behavior tests (real overlay, runtime)
# --------------------------------------------------------------------------- #
def behavior(overlay: str) -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(overlay, trust_remote_code=True)
    res = {"resolved_class": type(tok).__name__, "is_fast": tok.is_fast, "vectors": {}}
    bad = []
    for name, s in VECTORS.items():
        ids = tok.encode(s)
        dec = tok.decode(ids, skip_special_tokens=True)
        rt = dec.strip() == s.strip()
        art = ("Ġ" in dec or "Ċ" in dec)
        kr_lost = any(ord(c) > 0x3000 for c in s) and not any(ord(c) > 0x3000 for c in dec)
        res["vectors"][name] = {"n_ids": len(ids), "roundtrip_ok": rt, "artifact": art, "korean_lost": kr_lost}
        if not rt or art or kr_lost:
            bad.append(name)
    try:
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": "대한민국의 수도는?"}], tokenize=False, add_generation_prompt=True)
        res["chat_template"] = {"ok": True, "korean_present": "수도" in rendered,
                                "artifact": ("Ġ" in rendered or "Ċ" in rendered)}
        if not res["chat_template"]["korean_present"] or res["chat_template"]["artifact"]:
            bad.append("chat_template")
    except Exception as e:
        res["chat_template"] = {"ok": f"ERR {e}"}; bad.append("chat_template")
    res["special_tokens"] = {"bos": tok.bos_token, "eos": tok.eos_token}
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if not tok.is_fast:
        bad.append("not_fast")
    print("BEHAVIOR_PASS" if not bad else f"BEHAVIOR_FAIL: {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--behavior":
        sys.exit(behavior(args[1]))
    sys.exit(unit())
