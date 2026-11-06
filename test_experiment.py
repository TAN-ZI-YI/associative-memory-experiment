"""Check baseline/control semantics without asserting a learned score."""
import unittest
import torch
from experiment import dictionary_answers, evaluate
from memory import DeltaBindingMemory, SymbolWorld, digest_model


class InterventionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_dictionary_exposes_exact_match_baseline_and_recent_window_failure(self):
        case = SymbolWorld(123).sample(4, heldout=True, counterfactual=True)
        answers = dictionary_answers(case["events"], case["contrast_cues"])
        self.assertTrue(torch.equal(answers, case["contrast_targets"] + 1))
        recent = dictionary_answers(case["events"], case["contrast_cues"], recent_only=True)
        self.assertTrue(torch.equal(recent, torch.zeros_like(recent)))

    def test_evaluation_preserves_parameters_and_scores_donor_target_separately(self):
        torch.manual_seed(7)
        model = DeltaBindingMemory().eval()
        digest = digest_model(model)
        report = evaluate(model, seed=7, sequences=4)
        self.assertEqual(digest_model(model), digest)
        self.assertEqual(report["metrics"]["changed_queries_follow_both_histories"]["total"], 8)
        self.assertEqual(report["metrics"]["stale_reverse_link_retained"]["total"], 4)
        self.assertTrue(report["controls"]["query_does_not_write_state"])
        self.assertEqual(report["controls"]["owner_source_free_read_max_difference"], 0)
        for example in report["examples"]:
            self.assertEqual(example["expected_a"][2], example["expected_b"][2])
            self.assertNotEqual(example["expected_a"][:2], example["expected_b"][:2])


if __name__ == "__main__":
    unittest.main()
