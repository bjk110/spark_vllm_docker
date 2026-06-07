# Release Management Notes

This document collects maintainer-oriented notes for Git tags, GHCR image
tags, and archived branch references.

It is not required for normal users who only want to run the published
containers.

## Git tags and GHCR image tags

GHCR image tags and Git tags do not necessarily move in lockstep — only
`v018-ngc2603` currently exists as a Git tag.

For user-facing image and preset mapping, see [`docs/images.md`](images.md).

## Recommended Git tags to create

> **Maintainer-only.** Do not run these commands blindly: verify the commit
> SHAs against the current `git log` first, confirm the corresponding GHCR
> image exists, and do not overwrite existing tags.

The maintainer can create the following tags to align Git tags with GHCR
image tags. Run from a clean checkout of `main`.

    git tag -a v019-ngc2603 7736716 -m "v019-ngc2603 — Gemma 4 + vLLM 0.19.1"
    git tag -a v020-ngc2603 8efdf0b -m "v020-ngc2603 — base-refresh-20260417 (vLLM 978a4462, FlashInfer 0.6.8)"
    git tag -a v021-ngc2603 8623187 -m "v021-ngc2603 — vLLM 95995bbe + FlashInfer v0.6.9"
    git tag -a v021-tq      3070f9a -m "v021-tq — base + TurboQuant cherry-picks + codegen workaround"
    git push origin v019-ngc2603 v020-ngc2603 v021-ngc2603 v021-tq

**Verify commit before tagging.** The four SHAs above were extracted from
`git log --oneline` at the time this document was last updated; if subsequent
work reshuffles `main`, re-locate the boundary commits with:

    git log --oneline --grep='base.refresh\|bump base.*v021\|0.19.1\|use Inductor graph partition'

Before creating tags:

1. Verify the commit SHA.
2. Confirm the corresponding GHCR image exists.
3. Confirm the image digest if reproducibility matters.
4. Do not overwrite existing tags.

## Branch structure

`main` is the only long-lived branch. All previously separate work
streams (base stack refresh, TurboQuant rebase, single-Spark CLUSTER_MODE,
unholy-fusion integration) have been merged in and their feature branches
deleted.

## Archived branch history

The legacy TurboQuant branch is preserved as a tag for reference:

- **`archive/feat-turboquant`**

If needed, it can be restored with:

```bash
git checkout -b feat/turboquant archive/feat-turboquant
```
