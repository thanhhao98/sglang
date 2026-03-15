import unittest

import torch

from sglang.srt.layers.attention.dcp_layout import build_dcp_local_page_table


class TestDcpLayout(unittest.TestCase):
    def test_token_table_uses_token_positions_not_page_columns(self):
        # One column per token position, as used by eager decode.
        global_page_table = torch.tensor(
            [[128 + i for i in range(129)]], dtype=torch.int32
        )
        full_seqlens = torch.tensor([129], dtype=torch.int32)

        local_page_table, local_seqlens = build_dcp_local_page_table(
            global_page_table=global_page_table,
            full_seqlens=full_seqlens,
            max_seq_len_k=129,
            page_size=64,
            dcp_rank=0,
            dcp_size=2,
        )

        self.assertEqual(local_seqlens.tolist(), [65])
        self.assertEqual(local_page_table.tolist(), [[128, 256]])

    def test_page_table_uses_page_columns(self):
        # One column per page, as used by CUDA-graph replay.
        global_page_table = torch.tensor([[2, 3, 4]], dtype=torch.int32)
        full_seqlens = torch.tensor([129], dtype=torch.int32)

        local_page_table, local_seqlens = build_dcp_local_page_table(
            global_page_table=global_page_table,
            full_seqlens=full_seqlens,
            max_seq_len_k=129,
            page_size=64,
            dcp_rank=0,
            dcp_size=2,
        )

        self.assertEqual(local_seqlens.tolist(), [65])
        self.assertEqual(local_page_table.tolist(), [[1, 2]])

    def test_negative_placeholders_stay_negative(self):
        global_page_table = torch.tensor([[10, -1, 12, -1, 14]], dtype=torch.int32)
        full_seqlens = torch.tensor([5], dtype=torch.int32)

        local_page_table, local_seqlens = build_dcp_local_page_table(
            global_page_table=global_page_table,
            full_seqlens=full_seqlens,
            max_seq_len_k=5,
            page_size=1,
            dcp_rank=1,
            dcp_size=2,
        )

        self.assertEqual(local_seqlens.tolist(), [2])
        self.assertEqual(local_page_table.tolist(), [[-1, -1]])


if __name__ == "__main__":
    unittest.main()
