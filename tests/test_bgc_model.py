import unittest
import torch

from fungidna.model.config import FungiDNAConfig
from fungidna.model.heads import MeanPoolHead
from fungidna.model.fungi_dna import FungiDNAForBGCBoundary


class TestBGCModel(unittest.TestCase):
    def test_mean_pool_head_output(self):
        head = MeanPoolHead(768, 2)
        x = torch.zeros(2, 5, 768)  # batch, seq(含首token), hidden
        out = head(x)
        self.assertEqual(out.shape, (2, 2))

    def test_bgc_boundary_forward_shape(self):
        # The StripedMambaBackbone uses mamba_ssm / flash_attn kernels that are
        # CUDA-only and fp16/bf16-only, so the real forward runs on cuda+bf16
        # (same invocation as the repo's training/inference scripts). On a
        # machine without CUDA the backbone forward cannot execute at all.
        if not torch.cuda.is_available():
            self.skipTest("StripedMambaBackbone (mamba_ssm/flash_attn) requires CUDA")
        model = FungiDNAForBGCBoundary(FungiDNAConfig(), num_outputs=2).cuda().bfloat16()
        ids = torch.zeros(2, 64, dtype=torch.long, device="cuda")
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(ids)
        self.assertEqual(out.shape, (2, 2))
