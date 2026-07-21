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
"""Draft-side DCP-off guard contract for DSparkWorkerV2.

The DSpark draft is a separate dense model with a REPLICATED (non
dcp-interleaved) KV pool, but the worker runs under the scheduler's ambient
ParallelContext where dcp_enabled is True. Any backend/pool code that
branches on ambient DCP (dcp-local lens, dcp page tables, sharded MLA cache
writes) would build sharded metadata against the unsharded draft pool — the
bug class behind a measured x0.83 accept-length regression on an EAGLE3
DCP branch (chain metadata built outside the draft guard).

These tests drive the REAL worker code with SimpleNamespace fakes under an
ambient ``get_parallel().override(dcp_enabled=True, dcp_size=8, dcp_rank=3)``
and assert ``get_parallel().dcp_enabled`` is False at every draft-side call
site — draft build/backends/graphs, the propose() draft forward, the
target-hidden KV injection, and the moe-draft idle forward — while the
NEGATIVE CONTROLS pin that target-verify sites still see DCP enabled.
"""

import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dspark_components import dspark_worker_v2 as worker_mod
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_AMBIENT = dict(dcp_enabled=True, dcp_size=8, dcp_rank=3)


class _SeqLens:
    """CPU stand-in for a cuda seq_lens tensor (record_stream is a no-op)."""

    def __init__(self, values):
        self._t = torch.tensor(values, dtype=torch.int64)

    def record_stream(self, stream):
        pass

    def __len__(self):
        return self._t.shape[0]


def _recorder(records, key, ret=None):
    def _fn(*args, **kwargs):
        records[key] = get_parallel().dcp_enabled
        return ret

    return _fn


def _bare_worker(**attrs):
    worker = object.__new__(DSparkWorkerV2)
    for name, value in attrs.items():
        object.__setattr__(worker, name, value)
    return worker


class TestDraftContextGuard(CustomTestCase):
    def test_draft_context_is_dcp_off(self):
        worker = _bare_worker(_draft_dp_context_enabled=False)
        with get_parallel().override(**_AMBIENT):
            self.assertTrue(get_parallel().dcp_enabled)
            with worker._draft_context():
                self.assertFalse(get_parallel().dcp_enabled)
                self.assertEqual(get_parallel().dcp_size, 1)
            self.assertTrue(get_parallel().dcp_enabled)

    def test_draft_context_composes_dp_tp_context(self):
        entered = {}

        @contextmanager
        def fake_tp_ctx(group):
            entered["group"] = group
            entered["dcp_on_entry"] = get_parallel().dcp_enabled
            yield

        worker = _bare_worker(_draft_dp_context_enabled=True)
        with (
            mock.patch.object(worker_mod, "draft_tp_context", fake_tp_ctx),
            get_parallel().override(attn_tp_group="fake-group", **_AMBIENT),
        ):
            with worker._draft_context():
                self.assertFalse(get_parallel().dcp_enabled)
        self.assertEqual(entered["group"], "fake-group")


class TestDraftSideCallSitesGuarded(CustomTestCase):
    def test_init_attention_backends_guarded(self):
        records = {}
        worker = _bare_worker(
            _draft_dp_context_enabled=False,
            _draft_worker=SimpleNamespace(
                init_attention_backends=_recorder(records, "init_backends")
            ),
            model_runner=SimpleNamespace(
                model_config=None, attn_backend=SimpleNamespace()
            ),
        )
        with (
            mock.patch.object(worker_mod, "mambaish_config", return_value=None),
            get_parallel().override(**_AMBIENT),
        ):
            worker.init_attention_backends()
        self.assertIs(records["init_backends"], False)

    def test_init_cuda_graphs_guarded(self):
        records = {}
        worker = _bare_worker(
            _draft_dp_context_enabled=False,
            server_args=SimpleNamespace(disable_cuda_graph=True),
            _draft_worker=SimpleNamespace(
                init_cuda_graphs=_recorder(records, "init_graphs")
            ),
        )
        with get_parallel().override(**_AMBIENT):
            worker.init_cuda_graphs()
        self.assertIs(records["init_graphs"], False)

    def test_decode_propose_guarded_target_verify_not(self):
        """The money test: in one real _forward_decode pass, the draft
        propose() must see DCP off while the target-verify executor and the
        accept path (negative controls) still see DCP on."""
        records = {}
        bs = 2
        gamma = 7
        draft_block_ids = torch.zeros((bs, 1), dtype=torch.int64)
        draft_tokens = torch.zeros((bs, gamma), dtype=torch.int64)
        proposal = SimpleNamespace(
            draft_block_ids=draft_block_ids,
            draft_block=SimpleNamespace(draft_tokens=draft_tokens),
            confidence=torch.ones(bs),
            folded=False,
            draft_hidden=None,
            confidence_tap=None,
        )
        layout = SimpleNamespace(verify_lens=torch.full((bs,), 8, dtype=torch.int32))
        target_verify = SimpleNamespace(
            logits_output=SimpleNamespace(
                next_token_logits=torch.zeros(bs, 8), hidden_states=None
            ),
            can_run_cuda_graph=False,
        )
        accept = SimpleNamespace(
            new_seq_lens=torch.tensor([12, 13]),
            commit_lens=torch.tensor([2, 3], dtype=torch.int32),
            correct_len=torch.tensor([2, 3]),
            cap_trim_lens=torch.zeros(bs, dtype=torch.int32),
            bonus=torch.zeros(bs, dtype=torch.int64),
            out_tokens=torch.zeros((bs, 1), dtype=torch.int64),
        )
        worker = _bare_worker(
            device="cpu",
            verify_num_draft_tokens=gamma + 1,
            _block_pos_offsets=None,
            _draft_dp_context_enabled=False,
            _draft_is_moe=False,
            _simulate_acc_len=0.0,
            server_args=SimpleNamespace(enable_dp_attention=False),
            model_runner=SimpleNamespace(),
            _target_worker=SimpleNamespace(
                model_runner=SimpleNamespace(model="target-model")
            ),
            _observers=SimpleNamespace(
                begin_step=lambda: None,
                segment=lambda s: nullcontext(),
                observe_verify_step=lambda **kwargs: None,
            ),
            _proposer=SimpleNamespace(propose=_recorder(records, "propose", proposal)),
            _verify_planner=SimpleNamespace(
                resolve_verify_token_budget=lambda **kwargs: 16,
                schedule_layout=lambda **kwargs: layout,
                should_run_compact=lambda **kwargs: False,
            ),
            _verify_executor=SimpleNamespace(
                verify_epilogue=None,
                run_non_compact=_recorder(records, "run_non_compact", target_verify),
                accept_and_finalize=_recorder(records, "accept_and_finalize", accept),
                commit_hidden=_recorder(records, "commit_hidden"),
            ),
        )
        object.__setattr__(
            worker,
            "_commit_target_mamba_states_after_verify",
            lambda **kwargs: None,
        )
        object.__setattr__(worker, "_dp_verify_tier_num_tokens", lambda batch: None)
        batch = SimpleNamespace(
            spec_info=object.__new__(DFlashDraftInputV2),
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            seq_lens=_SeqLens([10, 10]),
            sampling_info=None,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            global_num_tokens=None,
            forward_iter=3,
            reqs=[],
            spec_verify_tier_num_tokens=16,
        )
        with (
            mock.patch.object(
                worker_mod, "alloc_verify_window", lambda **kwargs: SimpleNamespace()
            ),
            mock.patch.object(
                worker_mod, "prepare_mamba_track_for_verify", lambda b: None
            ),
            mock.patch.object(
                worker_mod,
                "make_next_draft_input",
                lambda **kwargs: SimpleNamespace(**kwargs),
            ),
            get_parallel().override(**_AMBIENT),
        ):
            result = worker._forward_decode(batch, on_publish=None)
        self.assertIs(
            records["propose"],
            False,
            "draft propose() ran with ambient dcp_enabled=True "
            "(draft guard missing on the decode path)",
        )
        # negative controls: target-side verify must still see DCP
        self.assertIs(records["run_non_compact"], True)
        self.assertIs(records["accept_and_finalize"], True)
        self.assertIs(records["commit_hidden"], True)
        self.assertEqual(int(result.speculative_num_draft_tokens), gamma + 1)

    def test_idle_moe_draft_participation_guarded(self):
        records = {}
        worker = _bare_worker(
            device="cpu",
            verify_num_draft_tokens=8,
            _draft_dp_context_enabled=False,
            _draft_is_moe=True,
            server_args=SimpleNamespace(enable_dp_attention=True),
            _observers=SimpleNamespace(note_idle_decode_step=lambda: None),
            _proposer=SimpleNamespace(
                run_idle_participation=_recorder(records, "idle_draft")
            ),
            _verify_executor=SimpleNamespace(
                run_idle_participation=_recorder(records, "idle_verify")
            ),
        )
        object.__setattr__(worker, "_idle_verify_ragged_layout", lambda batch: None)
        batch = SimpleNamespace(
            spec_info=object.__new__(DFlashDraftInputV2),
            forward_mode=SimpleNamespace(is_idle=lambda: True),
        )
        with (
            mock.patch.object(
                worker_mod,
                "make_next_draft_input",
                lambda **kwargs: SimpleNamespace(**kwargs),
            ),
            get_parallel().override(**_AMBIENT),
        ):
            worker._forward_decode(batch, on_publish=None)
        self.assertIs(records["idle_draft"], False)
        self.assertIs(records["idle_verify"], True)  # negative control

    def test_inject_target_hidden_guarded(self):
        records = {}
        bs = 2
        batch_output = SimpleNamespace(
            logits_output=SimpleNamespace(hidden_states=torch.zeros(7, 4)),
            next_token_ids=torch.zeros(bs, dtype=torch.int64),
        )
        worker = _bare_worker(
            device="cpu",
            _draft_dp_context_enabled=False,
            server_args=SimpleNamespace(enable_dp_attention=False),
            model_runner=SimpleNamespace(
                server_args=SimpleNamespace(attention_backend="torch_native")
            ),
            _target_worker=SimpleNamespace(
                forward_batch_generation=lambda batch, capture_hidden_mode: batch_output
            ),
            _kv_injector=SimpleNamespace(
                inject_target_hidden=_recorder(records, "inject")
            ),
        )
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            seq_lens=torch.tensor([3, 4], dtype=torch.int64),
            extend_lens=[3, 4],
            prefix_lens=[0, 0],
            out_cache_loc=torch.arange(7, dtype=torch.int64),
        )
        with (
            mock.patch.object(
                worker_mod,
                "make_next_draft_input",
                lambda **kwargs: SimpleNamespace(**kwargs),
            ),
            mock.patch.object(
                worker_mod,
                "compute_position",
                lambda *args: (torch.zeros(7, dtype=torch.int64), None),
            ),
            get_parallel().override(**_AMBIENT),
        ):
            worker._forward_prefill(batch, on_publish=None)
        self.assertIs(
            records["inject"],
            False,
            "target-hidden KV injection into the replicated draft pool ran "
            "with ambient dcp_enabled=True",
        )


if __name__ == "__main__":
    unittest.main()
