"""Compare the independent audit arithmetic with PyTorch's numerical behavior."""

import unittest

import numpy as np
import torch

from audit_v8_strength import bf16_round, softmax64


class ArithmeticTests(unittest.TestCase):
    def test_bfloat16_rounding_including_halfway_values(self):
        values=np.random.default_rng(9).normal(size=10000).astype(np.float32)*16
        halfway=np.asarray([0x3f808000,0x3f818000,0xbf808000,0xbf818000,0,0x80000000],np.uint32).view(np.float32)
        values=np.r_[values,halfway]
        expected=torch.from_numpy(values).to(torch.bfloat16).float().numpy()
        np.testing.assert_array_equal(bf16_round(values).view(np.uint32),expected.view(np.uint32))

    def test_bfloat16_addition_and_probability_reconstruction(self):
        rng=np.random.default_rng(3)
        original=torch.from_numpy(rng.normal(size=(100,32)).astype(np.float32)).to(torch.bfloat16)
        bias=torch.from_numpy(rng.normal(size=(100,32)).astype(np.float32)*5).to(torch.bfloat16)
        observed=(original+bias).float().numpy()
        np.testing.assert_array_equal(bf16_round(original.float().numpy()+bias.float().numpy()),observed)
        np.testing.assert_allclose(softmax64(observed),torch.from_numpy(observed).softmax(-1).numpy(),atol=2e-7,rtol=2e-6)


if __name__=="__main__":
    torch.set_num_threads(1)
    unittest.main()
