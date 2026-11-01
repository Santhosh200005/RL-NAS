import unittest, numpy as np
from rlnas.search_space import *
from rlnas.reward import compute_reward
from rlnas.evaluators import SurrogateEvaluator
from rlnas.search import run_search
import yaml

CFG = yaml.safe_load(open("configs/smoke.yaml"))
try:
    import torch; HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestSpace(unittest.TestCase):
    def test_roundtrip_and_validity(self):
        rng = np.random.default_rng(0)
        for _ in range(500):
            a = sample_random(rng)
            validate(a)
            self.assertEqual(decode(encode_tokens(a)).key(), a.key())
            self.assertEqual(len(features(a)), N_TOKENS)

    def test_pool_repair(self):
        t = [3, 0, 0] + [0, 0, 1] * 5          # depth 5, every layer max-pool
        self.assertLessEqual(sum(p != "none" for _, _, p in decode(t).layers), MAX_POOLS)

    def test_complexity_reference(self):
        p, m = complexity(REFERENCE)
        self.assertEqual(p, (3*32*9+64) + (32*64*9+128) + (64*128*9+256) + 128*10+10)
        self.assertGreater(m, 0)

    def test_reward_penalises_cost(self):
        r = {"lam_params": .03, "lam_flops": .03}
        self.assertGreater(compute_reward(.8, 1e5, 1e6, 0, r), compute_reward(.8, 2e6, 1e8, 0, r))
        self.assertEqual(compute_reward(.8, 9e9, 9e9, 0, {}), .8)


class TestSearch(unittest.TestCase):
    def test_baselines_with_surrogate(self):
        for m in ("random", "bayes"):
            h = run_search(m, SurrogateEvaluator(), CFG, seed=0, budget=10)
            self.assertEqual(len(h), 10)
            self.assertTrue(all(h[i]["best_reward"] <= h[i+1]["best_reward"] for i in range(9)))

    def test_reproducible(self):
        a = run_search("random", SurrogateEvaluator(), CFG, 3, 8)
        b = run_search("random", SurrogateEvaluator(), CFG, 3, 8)
        self.assertEqual([r["key"] for r in a], [r["key"] for r in b])


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestTorch(unittest.TestCase):
    def test_param_count_matches_analytic(self):
        from rlnas.models import CandidateCNN
        rng = np.random.default_rng(1)
        for _ in range(20):
            a = sample_random(rng)
            n = sum(p.numel() for p in CandidateCNN(a).parameters())
            self.assertEqual(n, complexity(a)[0])
            self.assertEqual(CandidateCNN(a)(torch.zeros(2, 3, 32, 32)).shape, (2, 10))

    def test_controller_and_rl_loop(self):
        h = run_search("rl", SurrogateEvaluator(), CFG, seed=0, budget=12)
        self.assertEqual(len(h), 12)

    def test_real_training_smoke(self):
        from rlnas.evaluators import TorchEvaluator
        m = TorchEvaluator(CFG).evaluate(REFERENCE)
        self.assertGreater(m["val_acc"], 0.1)


if __name__ == "__main__":
    unittest.main()
