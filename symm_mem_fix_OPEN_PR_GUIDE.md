# Step-by-step: opening the symm-mem + CUDA graph fix PR

This is the operational checklist for opening the upstream PR for the
`htphan/fix-symm-mem-cuda-graph-deadlock` branch (already pushed to
`origin3 = git@github.com:thanhhao98/sglang.git`).

The PR description content lives in `symm_mem_fix_PR_DESCRIPTION.md`.

## 0. Branch state at this point

* **Local worktree:** `/Users/htphan/workspace/DLAlgo/sglang-fix-symmem`
* **Branch:** `htphan/fix-symm-mem-cuda-graph-deadlock`
* **Base:** `origin/main` (sgl-project/sglang) at commit `0be6ab04d` at the time of the fix commit
* **Pushed to:** `origin3` (thanhhao98/sglang fork)
* **Commits on top of upstream `main`:**
  1. `095526676` — `fix: disable symm-mem during CUDA graph capture to avoid NCCL deadlock`
  2. `cc4da65ad` — `test: add unit tests for ModelRunner._disable_symm_mem`
  3. `2a2e438f9` — `test: trim _disable_symm_mem unit tests to real-logic-only + use CustomTestCase`
* **Files touched (2):**
  * `python/sglang/srt/model_executor/model_runner.py` (+53 / −13)
  * `test/registered/unit/model_executor/test_disable_symm_mem.py` (+85, new)

> Note on history: commit 2 was kept (rather than amending) because it
> was already on `origin3`. Before opening the PR, consider squashing
> commits 2 and 3 into a single `test:` commit for a cleaner upstream
> history (see "interactive squash" below) — or leave the two commits
> if you'd like reviewers to see the trim diff explicitly.

```bash
# Optional: squash the two test commits into one before pushing.
cd /Users/htphan/workspace/DLAlgo/sglang-fix-symmem
git rebase -i origin/main
# In the editor, change "pick 2a2e438f9" to "fixup 2a2e438f9"
# Save & exit; then:
git push --force-with-lease origin3 htphan/fix-symm-mem-cuda-graph-deadlock
```

Verify with:

```bash
cd /Users/htphan/workspace/DLAlgo/sglang-fix-symmem
git log --oneline origin/main..HEAD
git diff --stat origin/main..HEAD
```

## 1. Pre-flight: pre-commit hooks

SGLang enforces a pre-commit suite (`isort`, `ruff F401,F821`, `black`,
`codespell`, `clang-format`, plus a few local hooks). Already verified
clean for this branch, but re-run before opening the PR to be safe:

```bash
cd /Users/htphan/workspace/DLAlgo/sglang-fix-symmem
pip3 install pre-commit
pre-commit install
pre-commit run --all-files
# If anything auto-fixed, re-run until clean and amend / add new commit.
```

## 2. Pre-flight: rebase onto latest upstream `main`

A long-standing branch can drift; rebase onto the freshest upstream `main`
just before opening the PR so the PR diff is minimal:

```bash
cd /Users/htphan/workspace/DLAlgo/sglang-fix-symmem
git fetch origin main
git rebase origin/main
# Resolve any conflicts, re-run pre-commit, and force-push to origin3.
git push --force-with-lease origin3 htphan/fix-symm-mem-cuda-graph-deadlock
```

`--force-with-lease` is safer than `--force`: it refuses if anyone else
pushed to the remote branch in the meantime.

## 3. Pre-flight: run the unit tests one more time

CPU-only, fast (<5 s), no GPU required:

```bash
cd /Users/htphan/workspace/DLAlgo/sglang-fix-symmem
python3 test/registered/unit/model_executor/test_disable_symm_mem.py
# or:
pytest test/registered/unit/model_executor/test_disable_symm_mem.py -v
```

## 4. Pre-flight: run the live repro on B200

Already verified on `colossus_b200_1` with `lmsysorg/sglang:latest`:

```bash
ssh colossus_b200_1
docker exec sglang-bench-fix bash -c '
  cd /sgl-workspace/sglang_fix && \
  PYTHONPATH=/sgl-workspace/sglang_fix/python \
  python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --tp 8 --attention-backend flashinfer --enable-symm-mem \
    --port 30000
'
# In another shell:
docker exec sglang-bench-fix bash -c '
  python3 -m sglang.test.few_shot_gsm8k --num-questions 50 --parallel 32 \
    --max-new-tokens 256 --port 30000
'
```

Expected (already confirmed):

* `Capture cuda graph end. ... mem usage=2.52 GB.` — graph capture finishes (no hang at `bs=144`).
* `INFO: Application startup complete.` — server is ready on port 30000.
* `Accuracy: 0.940` — GSM8K passes.

If anything regresses, fix it in a new commit, run pre-commit, push.

## 5. Open the PR via GitHub UI

The branch on the fork has a one-click "Compare & pull request" banner:
https://github.com/thanhhao98/sglang/pull/new/htphan/fix-symm-mem-cuda-graph-deadlock

In the PR creation form:

1. **Base repo / branch:** `sgl-project/sglang : main`
2. **Head repo / branch:** `thanhhao98/sglang : htphan/fix-symm-mem-cuda-graph-deadlock`
3. **Title:** `[Bug Fix] Disable symm-mem during CUDA graph capture to avoid NCCL deadlock`
4. **Body:** paste the contents of `symm_mem_fix_PR_DESCRIPTION.md` verbatim. The template sections (Motivation / Modifications / Accuracy Tests / Speed Tests / Checklist / Review and Merge Process) match the upstream PR template.
5. **Assignees / reviewers:** leave empty — Merge Oncalls and CODEOWNERS will be auto-tagged based on the touched paths (`model_executor/`).

Click **"Create pull request"**.

## 6. Open the PR via `gh` CLI (alternative)

If you prefer `gh`:

```bash
cd /Users/htphan/workspace/DLAlgo/sglang-fix-symmem
gh pr create \
  --repo sgl-project/sglang \
  --base main \
  --head thanhhao98:htphan/fix-symm-mem-cuda-graph-deadlock \
  --title "[Bug Fix] Disable symm-mem during CUDA graph capture to avoid NCCL deadlock" \
  --body-file /Users/htphan/workspace/DLAlgo/sglang/symm_mem_fix_PR_DESCRIPTION.md
```

`gh` will print the PR URL on success.

## 7. Post-open: get CI running

By default, PRs from forks do **not** auto-run CI. You need either an
authorized contributor to add the `run-ci` label, or to ping someone in
[`CI_PERMISSIONS.json`](https://github.com/sgl-project/sglang/blob/main/.github/CI_PERMISSIONS.json)
to comment one of:

* `/tag-run-ci-label` — adds the label, every future push triggers CI
* `/tag-and-rerun-ci` — adds the label and runs CI immediately

As the PR author you can always use `/rerun-failed-ci` on your own PR
even without being in `CI_PERMISSIONS.json`.

The relevant CI stages this PR will exercise:

* **stage-a-test-cpu** — picks up the new
  `test/registered/unit/model_executor/test_disable_symm_mem.py` (registered
  with `est_time=2`, `suite="stage-a-test-cpu"`).
* **stage-b-test-* / stage-c-test-*** — exercise the model_runner code
  path on real hardware. Symm-mem is gated by `--enable-symm-mem` which
  is off by default in the CI matrix, but any cuda-graph capture test
  validates the wrapping does not break the default path.

## 8. Post-open: address review comments

If reviewers ask for changes:

1. Edit on `htphan/fix-symm-mem-cuda-graph-deadlock` in the local
   worktree (`/Users/htphan/workspace/DLAlgo/sglang-fix-symmem`).
2. Run `pre-commit run --all-files` again.
3. Run the unit test again.
4. `git commit` (do **not** amend; keep the review history visible) and
   `git push origin3 htphan/fix-symm-mem-cuda-graph-deadlock`.
5. Re-trigger CI with `/rerun-failed-ci` (or just push — it auto-triggers
   if the `run-ci` label is on).

## 9. Post-merge cleanup

After the PR is merged into `sgl-project/sglang:main`:

```bash
cd /Users/htphan/workspace/DLAlgo/sglang-fix-symmem
git fetch origin
git checkout main && git pull
# Optional: delete the worktree once you're done with it.
cd /Users/htphan/workspace/DLAlgo
git -C sglang worktree remove sglang-fix-symmem
git -C sglang push origin3 --delete htphan/fix-symm-mem-cuda-graph-deadlock
```
