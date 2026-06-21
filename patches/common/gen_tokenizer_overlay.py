#!/usr/bin/env python3
"""
Generate a non-mutating tokenizer overlay for stepfun-ai/Step-3.7-Flash-FP8.

Why this exists
---------------
The checkpoint ships `tokenizer_config.json` with
`"tokenizer_class": "LlamaTokenizerFast"`. Under Transformers 5.10.2 that class
resolves to the SentencePiece-backed *slow* `LlamaTokenizer`, which drops
non-ASCII (Korean) text and collapses whitespace during encode -- e.g.
`"대한민국의 수도는 서울입니다."` encodes to 2 ids and decodes to `"."`. Setting
`"tokenizer_class": "PreTrainedTokenizerFast"` makes Transformers use the Rust
fast tokenizer (reads `tokenizer.json` directly), which is correct.

Instead of editing the synchronized read-only model directory in place, this
script builds an overlay directory containing only the small tokenizer
metadata, with the one field changed, and leaves the model mount untouched.
vLLM is then pointed at the overlay via `--tokenizer <overlay>`; the Step-3.7
multimodal processor reuses that same tokenizer (it is constructed as
`Step3VLProcessor(tokenizer=self.get_tokenizer())`, verified in vLLM 0.23
`model_executor/models/step3_vl.py`, inherited by `step3p7.py`), so both the
serving tokenizer and the VLM processor use the overlay.

Determinism / guards: see VALIDATED spec and generate(). The model source
directory is never modified; output is atomic; the script is idempotent and
fails non-zero on any mismatch. Paths are restricted (no writing into a model
root). The core is parameterized via a Spec so it can be unit-tested offline
with synthetic fixtures; main() always uses the VALIDATED spec.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Spec:
    src_tokcfg_sha: str
    out_tokcfg_sha: str
    expected_orig_class: str
    target_class: str
    verbatim: dict           # filename -> expected source sha256
    model_revision: str = ""
    allowed_model_roots: tuple = ("/models", "/home/bjk110/Documents/Models", "/mnt/data/llm-models")
    allowed_out_roots: tuple = ("/run/", "/tmp/", "/dev/shm/")
    tokcfg_name: str = "tokenizer_config.json"


# --- validated spec (Step-3.7-Flash-FP8, revision b3d7916…) ---
VALIDATED = Spec(
    src_tokcfg_sha="78202af487f4d4360e8d15cb0506d60718f8599770599c6f28f7f1fa045a591f",
    out_tokcfg_sha="e4bec1b1841cdb9da779f34c2260604ed27800252a445e7ad2811e3b37acc4ea",
    expected_orig_class="LlamaTokenizerFast",
    target_class="PreTrainedTokenizerFast",
    verbatim={
        "tokenizer.json": "b564c620eb77fa11d0926011c2202347d6cfc358d79724ee04ae7007e13636f0",
        "special_tokens_map.json": "d47424bda11df4cedc3f9458915c465a28e601d3b7df0e78f6dff4d7727006c4",
        "chat_template.jinja": "f428623fc81c940c35be3509fbffc086b4b4360d8800e46103e6f34d02891633",
    },
    model_revision="b3d7916fccac844cca050d7520f2aaa513f9a84f",
)


class OverlayError(RuntimeError):
    pass


def sha256_file(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def generate(model: str, out: str, spec: Spec = VALIDATED) -> dict:
    """Build the overlay. Returns the manifest dict. Raises OverlayError on any
    precondition/postcondition failure. Never modifies `model`."""
    model = os.path.realpath(model)
    out = os.path.realpath(out)
    TOKCFG = spec.tokcfg_name

    # --- path restrictions ---
    if not any(model == r or model.startswith(r.rstrip("/") + "/") for r in spec.allowed_model_roots):
        raise OverlayError(f"model path {model} not under an allowed root")
    if not any(out.startswith(r) for r in spec.allowed_out_roots):
        raise OverlayError(f"overlay path {out} not under an allowed ephemeral root")
    for r in spec.allowed_model_roots:
        if out == r or out.startswith(r.rstrip("/") + "/"):
            raise OverlayError("overlay path must not be inside a model root")

    src_cfg = os.path.join(model, TOKCFG)
    if not os.path.isfile(src_cfg):
        raise OverlayError(f"source {TOKCFG} not found at {src_cfg}")

    # --- precondition: source hash + class + JSON validity ---
    src_sha = sha256_file(src_cfg)
    if src_sha != spec.src_tokcfg_sha:
        raise OverlayError(f"source {TOKCFG} sha {src_sha} != expected upstream {spec.src_tokcfg_sha}")
    try:
        cfg = json.load(open(src_cfg))
    except json.JSONDecodeError as e:
        raise OverlayError(f"source {TOKCFG} is not valid JSON: {e}")
    if cfg.get("tokenizer_class") != spec.expected_orig_class:
        raise OverlayError(f"source tokenizer_class {cfg.get('tokenizer_class')!r} != {spec.expected_orig_class!r}")

    # --- precondition: verbatim files exist and match ---
    for fn, want in spec.verbatim.items():
        sp = os.path.join(model, fn)
        if not os.path.isfile(sp):
            raise OverlayError(f"required source file missing: {fn}")
        if sha256_file(sp) != want:
            raise OverlayError(f"source {fn} sha mismatch")

    # --- idempotency: existing valid overlay ---
    manifest_path = os.path.join(out, "overlay_manifest.json")
    if os.path.isfile(manifest_path):
        try:
            m = json.load(open(manifest_path))
            ok = (m.get("out_tokenizer_config_sha256") == spec.out_tokcfg_sha
                  and sha256_file(os.path.join(out, TOKCFG)) == spec.out_tokcfg_sha
                  and all(sha256_file(os.path.join(out, fn)) == w for fn, w in spec.verbatim.items()))
            if ok:
                return m
        except Exception:
            pass
        shutil.rmtree(out, ignore_errors=True)  # replace own stale overlay

    # --- reject an unexpected pre-existing path we do not own ---
    # At this point `out` should not exist: either it never did, or it was a
    # stale overlay (had a manifest) that we just removed. Anything still here
    # is a foreign directory/file with no overlay manifest -- refuse to clobber
    # it (os.replace would silently consume an empty dir or crash on a full one).
    if os.path.exists(out):
        raise OverlayError(f"refusing to overwrite unexpected existing path not owned by overlay: {out}")

    # --- build into temp dir, atomic rename ---
    parent = os.path.dirname(out)
    os.makedirs(parent, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=".overlay.tmp.", dir=parent)
    try:
        for fn in spec.verbatim:
            shutil.copy(os.path.join(model, fn), os.path.join(tmp, fn))
        cfg2 = json.load(open(src_cfg))
        cfg2["tokenizer_class"] = spec.target_class
        out_cfg_text = json.dumps(cfg2, ensure_ascii=False, indent=2)  # no trailing newline
        open(os.path.join(tmp, TOKCFG), "w").write(out_cfg_text)

        # --- postconditions ---
        out_sha = hashlib.sha256(out_cfg_text.encode()).hexdigest()
        if out_sha != spec.out_tokcfg_sha:
            raise OverlayError(f"regenerated {TOKCFG} sha {out_sha} != validated postimage {spec.out_tokcfg_sha}")
        a, b = json.load(open(src_cfg)), json.load(open(os.path.join(tmp, TOKCFG)))
        diffk = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        if diffk != ["tokenizer_class"]:
            raise OverlayError(f"unexpected changed fields: {diffk}")
        for fn, want in spec.verbatim.items():
            if sha256_file(os.path.join(tmp, fn)) != want:
                raise OverlayError(f"copied {fn} hash drift")
        if sha256_file(src_cfg) != spec.src_tokcfg_sha:
            raise OverlayError("source tokenizer_config changed during generation")

        manifest = {
            "model_source": model, "overlay_dir": out,
            "model_revision": spec.model_revision,
            "src_tokenizer_config_sha256": spec.src_tokcfg_sha,
            "out_tokenizer_config_sha256": spec.out_tokcfg_sha,
            "original_tokenizer_class": spec.expected_orig_class,
            "overlay_tokenizer_class": spec.target_class,
            "verbatim_files": dict(spec.verbatim),
            "files": sorted(list(spec.verbatim) + [TOKCFG, "overlay_manifest.json"]),
        }
        json.dump(manifest, open(os.path.join(tmp, "overlay_manifest.json"), "w"), indent=2)
        os.replace(tmp, out)
        tmp = None
        return manifest
    finally:
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="source model dir (read-only)")
    ap.add_argument("--out", required=True, help="overlay output dir (ephemeral)")
    ap.add_argument("--print-manifest", action="store_true")
    args = ap.parse_args()
    try:
        m = generate(args.model, args.out, VALIDATED)
    except OverlayError as e:
        print(f"[gen_tokenizer_overlay] FAIL: {e}", file=sys.stderr)
        return 1
    print(f"[gen_tokenizer_overlay] overlay OK at {args.out} "
          f"({VALIDATED.expected_orig_class} -> {VALIDATED.target_class}, "
          f"{VALIDATED.src_tokcfg_sha[:8]} -> {VALIDATED.out_tokcfg_sha[:8]}, source unchanged)")
    if args.print_manifest:
        print(json.dumps(m, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
