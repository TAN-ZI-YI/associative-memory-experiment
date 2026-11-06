"""Mechanical regressions for numeric state flow, not accuracy requirements."""

import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from memory import (
    DISTRACTORS, PAIRS, VOCABULARY, BindingStateOwner, DeltaBindingMemory, SymbolWorld,
)
from memory import digest_model


class BindingRetentionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        self.model = DeltaBindingMemory().eval()
        self.case = SymbolWorld(444).sample(4, heldout=True, counterfactual=True)

    def test_split_is_disjoint_and_counterfactual_keeps_counts_recent_inputs_and_cues(self):
        world = SymbolWorld(123)
        training = world.sample(32)
        for events, heldout in [(training["events"], False), (self.case["events"], True), (self.case["counterfactual_events"], True), (self.case["correction"], True)]:
            pairs = events[:, :PAIRS] if events.shape[1] > PAIRS else events
            for pair in pairs.flatten(0, 1):
                self.assertEqual(bool(world.heldout[pair[0] - 1, pair[1] - 1]), heldout)
        original = self.case["events"]
        changed = self.case["counterfactual_events"]
        self.assertTrue(torch.equal(original[:, :PAIRS].flatten(1).sort(-1).values, changed[:, :PAIRS].flatten(1).sort(-1).values))
        self.assertTrue(torch.equal(original[:, -DISTRACTORS:], changed[:, -DISTRACTORS:]))
        self.assertTrue(torch.equal(self.case["contrast_targets"][:, 2:], self.case["counterfactual_targets"][:, 2:]))
        self.assertFalse(torch.equal(self.case["contrast_targets"][:, :2], self.case["counterfactual_targets"][:, :2]))

    def test_target_and_cue_cannot_enter_event_formation(self):
        events = self.case["events"]
        first = self.model.write(events)
        self.case["targets"].fill_(0)
        self.case["cues"].fill_(1)
        self.assertTrue(torch.equal(first, self.model.write(events)))
        with self.assertRaises(TypeError):
            self.model.write(events, targets=self.case["targets"])
        with self.assertRaises(TypeError):
            self.model.write(events, cues=self.case["cues"])

    def test_single_masked_observation_loss_reaches_write_and_read_parameters(self):
        logits = self.model(self.case["events"], self.case["cues"])
        F.cross_entropy(logits.flatten(0, 1), self.case["targets"].flatten()).backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0, name)

    def test_event_token_order_has_no_answer_role(self):
        original = self.model.write(self.case["events"])
        reversed_tokens = self.model.write(self.case["events"].flip(-1))
        self.assertTrue(torch.allclose(original, reversed_tokens, atol=2e-6, rtol=2e-6))

    def test_source_free_owner_returns_copies_and_read_does_not_write(self):
        owner = BindingStateOwner("fictional-a", self.model)
        source = self.case["events"][0].clone()
        for index, event in enumerate(source):
            owner.observe(f"event-{index}", "fictional-a", event)
        snapshot = owner.snapshot()
        expected = owner.read("fictional-a", self.case["cues"][0])
        source.zero_()
        snapshot["state"].fill_(999)
        returned = owner.read("fictional-a", self.case["cues"][0])
        self.assertTrue(torch.equal(expected, returned))
        returned.fill_(999)
        self.assertTrue(torch.equal(expected, owner.read("fictional-a", self.case["cues"][0])))
        self.assertEqual(owner.version, PAIRS + DISTRACTORS)
        self.assertNotIn("events", owner.__dict__)
        self.assertNotIn("cues", owner.__dict__)

    def test_old_events_can_change_numeric_state_after_identical_recent_inputs(self):
        state = self.model.write(self.case["events"])
        other = self.model.write(self.case["counterfactual_events"])
        self.assertGreater(float((state - other).abs().max().detach()), 0)
        # This only checks an available route, not correctness of learned recall.

    def test_failed_write_preserves_state_version_and_event_retry(self):
        owner = BindingStateOwner("fictional-a", self.model)
        event = self.case["events"][0, 0]
        owner.observe("event-1", "fictional-a", event)
        old = owner.snapshot()["state"]
        with patch.object(self.model, "write", return_value=torch.full_like(old, float("nan"))):
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                owner.observe("event-2", "fictional-a", event)
        self.assertEqual(owner.version, 1)
        self.assertTrue(torch.equal(owner.snapshot()["state"], old))
        self.assertEqual(owner.observe("event-2", "fictional-a", event)["version"], 2)

    def test_subject_replay_invalid_input_and_changed_checkpoint_are_rejected(self):
        owner = BindingStateOwner("fictional-a", self.model)
        event = self.case["events"][0, 0]
        owner.observe("event-1", "fictional-a", event)
        for event_id, subject, symbols in [("event-1", "fictional-a", event), ("event-2", "fictional-b", event), ("event-2", "fictional-a", torch.zeros(2, dtype=torch.long)), ("event-2", "fictional-a", torch.tensor([1, VOCABULARY + 1])), ("event-2", "fictional-a", event.float())]:
            with self.assertRaises(ValueError):
                owner.observe(event_id, subject, symbols)
        self.assertEqual(owner.version, 1)
        with torch.no_grad():
            self.model.key_projection.weight[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "checkpoint changed"):
            owner.read("fictional-a", self.case["cues"][0])

    def test_initializations_and_cases_are_reproducible(self):
        torch.manual_seed(11)
        self.assertEqual(digest_model(self.model), digest_model(DeltaBindingMemory()))
        other = SymbolWorld(444).sample(4, heldout=True, counterfactual=True)
        for field, value in self.case.items():
            self.assertTrue(torch.equal(value, other[field]), field)


if __name__ == "__main__":
    unittest.main()
