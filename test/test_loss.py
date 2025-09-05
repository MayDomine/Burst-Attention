import unittest
import torch
import bmtrain as bmt
import torch.nn.functional as F
from burst_attn.sl_loss import OpSLFusedProjectionCrossEntropy
import math
import matplotlib.pyplot as plt


class TestOpSLFusedProjectionCrossEntropyConfig:
    def __init__(self, batch_size, sequence_length, num_splits, ignore_index, scale):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_splits = num_splits
        self.ignore_index = ignore_index
        self.scale = scale


class TestOpSLFusedProjectionCrossEntropy(unittest.TestCase):
    def generate_test_configs(self) -> list:
        configs = []
        for batch_size in [8, 16, 32]:
            for sequence_length in [10, 20, 50, 128]:
                for num_splits in [2, 4, 8, 16]:
                    for scale in [True, False]:
                        configs.append(
                            {
                                "batch_size": batch_size,
                                "sequence_length": sequence_length,
                                "num_splits": num_splits,
                                "scale": scale,
                            }
                        )
        return configs

    def ref_proj_loss(self, ignore_index, target, weight, x, scale):
        if scale:
            logits = F.linear(x / math.sqrt(weight.size(1)), weight)
        else:
            logits = F.linear(x, weight)
        target = target.to(dtype=torch.int64)
        ref_loss = torch.nn.functional.cross_entropy(
            logits.to(dtype=torch.float32),
            target,
            ignore_index=ignore_index,
            reduction="none",
        )
        return ref_loss

    def ref_fused_proj_loss(self, ignore_index, target, weight, x, scale):
        loss_fn = bmt.loss.FusedCrossEntropy(ignore_index=-100, reduction="none")
        if scale:
            logits = F.linear(x / math.sqrt(weight.size(1)), weight)
        else:
            logits = F.linear(x, weight)
        return loss_fn(logits, target)

    def ref_test(self, ignore_index, scale, target, weight, x, func) -> None:
        ref_weight = weight.clone().detach().requires_grad_(True)
        ref_inp = x.clone().detach().requires_grad_(True)
        ref_loss = func(ignore_index, target, ref_weight, ref_inp, scale)
        return ref_inp, ref_loss, ref_weight

    def forward_backward(self, config=None) -> None:
        if config is None:
            config = self.generate_test_configs()[0]

        batch_size = config["batch_size"]
        sequence_length = config["sequence_length"]
        num_splits = config["num_splits"]
        ignore_index = -100
        scale = config["scale"]
        num_classes = 10
        dtype = torch.float16
        hidden = 128

        device = "cuda"
        torch.cuda.set_device(0)
        x = torch.randn(
            batch_size * sequence_length,
            hidden,
            requires_grad=True,
            dtype=dtype,
            device=device,
        )
        weight = torch.randn(
            num_classes, hidden, requires_grad=True, dtype=dtype, device=device
        )
        target = torch.randint(
            0,
            num_classes - 1,
            (batch_size * sequence_length,),
            dtype=torch.int32,
            device=device,
        )

        loss_fn = OpSLFusedProjectionCrossEntropy.apply
        loss = loss_fn(x, weight, target, ignore_index, num_splits, scale)

        ref_inp, ref_loss, ref_weight = self.ref_test(
            ignore_index, scale, target, weight, x, self.ref_fused_proj_loss
        )
        self.assertIsInstance(loss, torch.Tensor)
        torch.testing.assert_close(loss, ref_loss)
        torch.testing.assert_close(loss.sum(), ref_loss.sum())
        loss.sum().backward()
        ref_loss.sum().backward()
        torch.testing.assert_close(x.grad, ref_inp.grad, rtol=1e-2, atol=1e-2)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(weight.grad)

        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(weight.grad.shape, weight.shape)

        self.assertTrue(torch.any(x.grad != 0))
        self.assertTrue(torch.any(weight.grad != 0))

    def test_forward(self):
        configs = self.generate_test_configs()
        for config in configs:
            self.forward_backward(config)


if __name__ == "__main__":
    test_case = TestOpSLFusedProjectionCrossEntropy()
    conf = test_case.generate_test_configs()[0]
    test_case.forward_backward(conf)
    # unittest.main()
