# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Kernel-contract canary for the single-call CP-causal DCP target verify.

The DCP verify path folds each rank's ENTIRE local interleaved slice
(committed prefix + the draft tokens this rank owns) in one TokenSpeed call
with per-request GLOBAL ``causal_seqs`` (= prefix + draft_token_num) and
DCP-local ``seq_lens``; the kernel must resolve each (request, q_tok) causal
bound in GLOBAL coordinates BEFORE the cp divide, and the cross-rank base-2
LSE merge of the partials must reproduce the unsharded result. The existing
unit tests pin the metadata arithmetic (``global_seq_lens_k``/``seq_lens_k``
split) with a mocked kernel; this test pins the KERNEL contract itself, going
through ``TokenspeedMLABackend._run_decode_kernel`` — the exact call
convention the serving path uses.

Two configurations, both asserted against an fp32 reference computed here:
  - CLEAN: random fp8 latents. PASS bar: cp-merged rel-fro within 3x of the
    cp1 (unsharded single-call) calibration floor, i.e. fp8 requant noise.
  - POISON: every draft row's latent set to +200 (fp8-representable,
    dominates softmax). The reference includes the same poison, so outputs
    match ONLY if the kernel's per-(request, q_tok) bound agrees with the
    reference on every poisoned row: any +-1 bound error explodes to O(1)
    rel error. This detects off-by-one masking DETERMINISTICALLY, not
    statistically.

q_len = 8 = DSpark verify width (gamma=7 draft block + bonus), the widest
shape the workspace bound (_TOKENSPEED_MAX_Q_LEN) admits.

Run manually:
    python -m pytest test/registered/dcp/test_dcp_single_call_verify.py -v
"""

import importlib.util
import math
import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-b200")

KV_LORA, ROPE = 512, 64
DIM = KV_LORA + ROPE
PAGE = 64
SCALE = 1.0 / math.sqrt(DIM)
T = 8  # DSpark verify_num_draft_tokens (gamma=7 + bonus)
HEADS = 16
PREFIXES = [997, 1531]  # distinct residues mod 4 AND mod 8, multi-page
_MIN_SM = 100


def _supported() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA is required"
    if importlib.util.find_spec("tokenspeed_mla") is None:
        return False, "tokenspeed_mla python package is not installed"
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    if sm < _MIN_SM:
        return (
            False,
            f"tokenspeed_mla requires SM {_MIN_SM // 10}.{_MIN_SM % 10}+ "
            f"(Blackwell), got SM {major}.{minor}",
        )
    return True, ""


_SUPPORTED, _SKIP_REASON = _supported()


def _make_backend(dev):
    import tokenspeed_mla

    from sglang.srt.layers.attention.tokenspeed_mla_backend import (
        TokenspeedMLABackend,
    )

    backend = object.__new__(TokenspeedMLABackend)
    backend.kv_lora_rank = KV_LORA
    backend.qk_rope_head_dim = ROPE
    backend._tokenspeed_workspace = torch.empty(
        tokenspeed_mla.get_num_sm(dev) * HEADS * T * (KV_LORA + 1) * 4,
        dtype=torch.int8,
        device=dev,
    )
    return backend


def _build_scene(prefixes, poison, dev, seed=1234):
    torch.manual_seed(seed)
    bs = len(prefixes)
    totals = [p + T for p in prefixes]
    q = torch.randn(bs, T, HEADS, DIM, device=dev).to(torch.float8_e4m3fn)
    kv_rows = []
    for i, tot in enumerate(totals):
        rows = torch.randn(tot, DIM, device=dev)
        if poison:
            # DISTINCT fp8-exact values per draft row (128 + 32j, all <= 448).
            # Identical poison rows would make bound-GROW errors nearly
            # invisible (one more identical row barely shifts the softmax,
            # measured ~0.8% instead of O(1)); distinct rows displace the
            # dominant visible row, so +-1 bound errors in EITHER direction
            # move the output O(1).
            vals = torch.arange(128.0, 128.0 + 32.0 * T, 32.0, device=dev)
            rows[prefixes[i] : tot] = vals.unsqueeze(1)
        kv_rows.append(rows.to(torch.float8_e4m3fn))
    return q, kv_rows, totals


def _fp32_reference(q, kv_rows, prefixes):
    outs = []
    for i, rows in enumerate(kv_rows):
        qf = q[i].float()
        kf = rows.float()
        vf = kf[:, :KV_LORA]
        toks = []
        for t in range(T):
            bound = prefixes[i] + t + 1  # exclusive end, self-inclusive rule
            logits = torch.einsum("hd,nd->hn", qf[t], kf[:bound]) * SCALE
            w = torch.softmax(logits, dim=-1)
            toks.append(torch.einsum("hn,nc->hc", w, vf[:bound]))
        outs.append(torch.stack(toks))
    return torch.stack(outs)  # [bs, T, H, KV_LORA]


def _paged_local(kv_rows, totals, cp_world, cp_rank, dev):
    """Rank-local rows per the owner rule pos % cp_world == cp_rank (the
    get_dcp_lens layout), packed into PAGE-row pages with a block table."""
    seqs, pools = [], []
    for rows, tot in zip(kv_rows, totals):
        owned = torch.arange(cp_rank, tot, cp_world, device=dev)
        local = rows[owned]
        n = local.shape[0]
        pages = max(1, math.ceil(n / PAGE))
        pool = torch.zeros(pages, PAGE, DIM, device=dev, dtype=torch.float8_e4m3fn)
        pool.view(-1, DIM)[:n] = local
        pools.append(pool)
        seqs.append(n)
    max_pages = max(p.shape[0] for p in pools)
    kv = torch.zeros(
        1 + sum(p.shape[0] for p in pools),
        PAGE,
        DIM,
        device=dev,
        dtype=torch.float8_e4m3fn,
    )
    bt = torch.zeros(len(pools), max_pages, dtype=torch.int32, device=dev)
    nxt = 1
    for i, pool in enumerate(pools):
        np_ = pool.shape[0]
        kv[nxt : nxt + np_] = pool
        bt[i, :np_] = torch.arange(nxt, nxt + np_, dtype=torch.int32, device=dev)
        nxt += np_
    return kv, bt, torch.tensor(seqs, dtype=torch.int32, device=dev)


def _call(backend, q, kv, bt, seq, max_seq, cp_world, cp_rank, causal_seqs):
    # layer.scaling * k_scale_float = softmax scale; output_scale = k_scale.
    layer = SimpleNamespace(scaling=SCALE, k_scale_float=1.0)
    return backend._run_decode_kernel(
        query=q,
        kv_cache=kv.unsqueeze(1),  # [pages, 1, PAGE, DIM] serving layout
        block_tables=bt,
        seq_lens=seq,
        max_seq_len=max(int(max_seq), 1),
        layer=layer,
        causal_seqs=causal_seqs,
        cp_world=cp_world,
        cp_rank=cp_rank,
        return_lse=True,
    )


def _lse_merge(outs, lses):
    """Cross-rank merge, base-2 LSE (dcp_a2a_lse_reduce is_lse_base_on_e=False)."""
    m = torch.stack(lses).max(0).values
    m = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    num = torch.zeros_like(outs[0])
    den = torch.zeros_like(lses[0])
    for o, l in zip(outs, lses):
        w = torch.exp2(l - m)
        w = torch.where(torch.isfinite(w), w, torch.zeros_like(w))
        num += o * w.unsqueeze(-1)
        den += w
    return num / den.clamp_min(1e-30).unsqueeze(-1)


@unittest.skipUnless(_SUPPORTED, _SKIP_REASON)
class TestDcpSingleCallVerify(CustomTestCase):

    def _run(self, cp_world, poison):
        dev = torch.device("cuda")
        backend = _make_backend(dev)
        q, kv_rows, totals = _build_scene(PREFIXES, poison, dev)
        ref = _fp32_reference(q, kv_rows, PREFIXES)

        # calibration: unsharded single call, same causal semantics (the
        # non-DCP verify call shape: causal_seqs=None, causality from q_len)
        kv1, bt1, seq1 = _paged_local(kv_rows, totals, 1, 0, dev)
        o1, _ = _call(backend, q, kv1, bt1, seq1, seq1.max().item(), 1, 0, None)
        floor = ((o1.reshape(ref.shape).float() - ref).norm() / ref.norm()).item()

        causal_seqs = torch.tensor(totals, dtype=torch.int32, device=dev)
        outs, lses = [], []
        for r in range(cp_world):
            kvr, btr, seqr = _paged_local(kv_rows, totals, cp_world, r, dev)
            o, l = _call(
                backend, q, kvr, btr, seqr, seqr.max().item(), cp_world, r, causal_seqs
            )
            outs.append(o.reshape(ref.shape).float())
            lses.append(l.reshape(ref.shape[:-1]).float())
        merged = _lse_merge(outs, lses)
        rel = ((merged - ref).norm() / ref.norm()).item()
        per_tok = (merged - ref).flatten(2).norm(dim=-1) / ref.flatten(2).norm(
            dim=-1
        ).clamp_min(1e-9)
        tol = 3 * max(floor, 1e-3)
        self.assertLessEqual(
            rel,
            tol,
            f"cp{cp_world} rel {rel:.4f} vs floor {floor:.4f} (poison={poison})",
        )
        self.assertLessEqual(
            per_tok.max().item(),
            3 * tol,
            f"worst-token rel {per_tok.max().item():.4f} (poison={poison}) — "
            "a per-(request,q_tok) causal-bound error",
        )

    def test_clean_matches_fp32_reference_cp4(self):
        self._run(cp_world=4, poison=False)

    def test_clean_matches_fp32_reference_cp8(self):
        self._run(cp_world=8, poison=False)

    def test_poisoned_draft_rows_pin_the_causal_bounds_cp4(self):
        self._run(cp_world=4, poison=True)

    def test_poisoned_draft_rows_pin_the_causal_bounds_cp8(self):
        self._run(cp_world=8, poison=True)


if __name__ == "__main__":
    unittest.main()
