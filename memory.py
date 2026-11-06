"""Learned associative memory for a finite symbolic state-intervention study."""

import hashlib

import torch
from torch import nn
from torch.nn import functional as F


def digest_model(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


VOCABULARY = 24
WIDTH = 32
DISTRACTORS = 16
PAIRS = 3


def validate_events(events):
    if not isinstance(events, torch.Tensor) or events.ndim != 3:
        raise ValueError("events must be [batch, events, two padded symbols]")
    if events.shape[0] < 1 or events.shape[1] < 1 or events.shape[2] != 2:
        raise ValueError("event shape mismatch")
    if events.dtype != torch.long or events.device.type != "cpu":
        raise ValueError("events must be CPU int64 symbols")
    if bool(((events < 0) | (events > VOCABULARY)).any()) or bool((events == 0).all(dim=-1).any()):
        raise ValueError("symbols out of range or empty observation")


def validate_cues(cues, batch):
    if not isinstance(cues, torch.Tensor) or cues.ndim != 2 or cues.shape[0] != batch or cues.shape[1] < 1:
        raise ValueError("cues must be [batch, queries]")
    if cues.dtype != torch.long or cues.device.type != "cpu" or bool(((cues < 1) | (cues > VOCABULARY)).any()):
        raise ValueError("cues must be nonpadding CPU int64 symbols")


class DeltaBindingMemory(nn.Module):
    """Event attention learns content, addresses and write strength.

    M' = M + sum_i beta_i * (v_i - M k_i) k_i^T.
    The delta rule and symbol-level write addresses are architectural priors.
    No expected binding matrix or cue is used to construct an event write.
    """

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCABULARY + 1, WIDTH, padding_idx=0)
        self.event_attention = nn.MultiheadAttention(WIDTH, 2, batch_first=True)
        self.key_projection = nn.Linear(WIDTH, WIDTH, bias=False)
        self.value_projection = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.Tanh(), nn.Linear(WIDTH, WIDTH))
        self.write_gate = nn.Linear(2 * WIDTH, 1)
        self.answer_projection = nn.Linear(WIDTH, VOCABULARY)

    def initial_state(self, batch):
        return torch.zeros(batch, WIDTH, WIDTH)

    def encode_events(self, events):
        validate_events(events)
        batch, length, _ = events.shape
        flat = events.reshape(-1, 2)
        embedded = self.embedding(flat)
        context, _ = self.event_attention(embedded, embedded, embedded, key_padding_mask=flat.eq(0), need_weights=False)
        features = torch.cat((embedded, context), dim=-1)
        keys = F.normalize(self.key_projection(embedded), dim=-1)
        values = self.value_projection(features)
        strengths = self.write_gate(features).sigmoid().squeeze(-1) * flat.ne(0)
        return (
            keys.reshape(batch, length, 2, WIDTH),
            values.reshape(batch, length, 2, WIDTH),
            strengths.reshape(batch, length, 2),
        )

    def update(self, state, keys, values, strengths):
        # Simultaneous writes make token order immaterial; no hand-written
        # rule selects the other token or designates a symbol's semantic role.
        predicted = torch.einsum("bvk,bik->biv", state, keys)
        residual = (values - predicted) * strengths[..., None]
        return state + torch.einsum("biv,bik->bvk", residual, keys)

    def write(self, events, state=None):
        keys, values, strengths = self.encode_events(events)
        if state is None:
            state = self.initial_state(events.shape[0])
        for index in range(events.shape[1]):
            state = self.update(state, keys[:, index], values[:, index], strengths[:, index])
        return state

    def read(self, state, cues):
        validate_cues(cues, state.shape[0])
        queries = F.normalize(self.key_projection(self.embedding(cues)), dim=-1)
        recalled = torch.einsum("bvk,bqk->bqv", state, queries)
        return self.answer_projection(recalled)

    def forward(self, events, cues):
        # Formation finishes before Cue is used. State carries the content.
        return self.read(self.write(events), cues)


class BindingStateOwner:
    """Single-subject in-memory state; retains no raw events or cues."""

    def __init__(self, subject_id, model):
        if not isinstance(subject_id, str) or not subject_id:
            raise ValueError("subject id required")
        self.subject_id = subject_id
        self.model = model
        self.checkpoint_sha256 = digest_model(model)
        self._state = model.initial_state(1)
        self._event_ids = set()
        self.version = 0

    def _check_checkpoint(self):
        if digest_model(self.model) != self.checkpoint_sha256:
            raise ValueError("checkpoint changed")

    @torch.no_grad()
    def observe(self, event_id, subject_id, symbols):
        self._check_checkpoint()
        if subject_id != self.subject_id:
            raise ValueError("subject mismatch")
        if not isinstance(event_id, str) or not event_id or event_id in self._event_ids:
            raise ValueError("event missing or replayed")
        if not isinstance(symbols, torch.Tensor) or symbols.shape != (2,):
            raise ValueError("one padded observation required")
        candidate = self.model.write(symbols.reshape(1, 1, 2), self._state)
        if candidate.shape != (1, WIDTH, WIDTH) or not torch.isfinite(candidate).all():
            raise ValueError("nonfinite or invalid derived state")
        committed = candidate.detach().clone()
        receipt = {"version": self.version + 1, "event_id": event_id, "subject_id": self.subject_id, "checkpoint_sha256": self.checkpoint_sha256}
        self._state = committed
        self._event_ids.add(event_id)
        self.version += 1
        return receipt

    @torch.no_grad()
    def read(self, subject_id, cues):
        self._check_checkpoint()
        if subject_id != self.subject_id:
            raise ValueError("subject mismatch")
        if not isinstance(cues, torch.Tensor) or cues.ndim != 1:
            raise ValueError("one subject's cues required")
        logits = self.model.read(self._state, cues[None])
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite readout")
        return logits[0].clone()

    def snapshot(self):
        self._check_checkpoint()
        return {"representation": "NUMERIC_DELTA_BINDING_STATE_V1", "subject_id": self.subject_id, "version": self.version, "checkpoint_sha256": self.checkpoint_sha256, "state": self._state.clone()}


class SymbolWorld:
    """Known symbols, disjoint train/test unordered pairs; labels stay here.

    Counterfactual histories have the same symbol counts and later inputs.
    Pair membership is synthetic experience content, not an input Bond label.
    """

    def __init__(self, seed):
        self.generator = torch.Generator().manual_seed(seed)
        split_rng = torch.Generator().manual_seed(20261007)
        random_edges = torch.rand(VOCABULARY, VOCABULARY, generator=split_rng) < .33
        self.heldout = torch.triu(random_edges, diagonal=1)
        self.heldout = self.heldout | self.heldout.t()
        self.split_sha256 = hashlib.sha256(self.heldout.numpy().tobytes()).hexdigest()

    def _identities(self, batch, heldout, counterfactual):
        accepted = []
        count = 0
        for _ in range(2000):
            proposed = torch.rand(max(1024 if counterfactual else 128, batch * 2), VOCABULARY, generator=self.generator).argsort(dim=-1)[:, :7]
            a, b, c, d, e, f, replacement = proposed.unbind(dim=-1)
            pairs = [(a, b), (c, d), (e, f)]
            if counterfactual:
                pairs += [(a, d), (c, b), (a, replacement)]
            valid = torch.ones(len(proposed), dtype=torch.bool)
            for left, right in pairs:
                valid &= self.heldout[left, right].eq(heldout)
            rows = proposed[valid]
            accepted.append(rows)
            count += len(rows)
            if count >= batch:
                return torch.cat(accepted)[:batch] + 1
        raise RuntimeError("unable to sample predefined pair split")

    def sample(self, batch, heldout=False, counterfactual=False):
        identities = self._identities(batch, heldout, counterfactual)
        pairs = identities[:, :6].reshape(batch, PAIRS, 2)
        swapped = pairs.clone()
        swapped[:, 0, 1], swapped[:, 1, 1] = pairs[:, 1, 1], pairs[:, 0, 1]
        # Randomize event order and each event's two token positions equally
        # in both histories, so positions cannot designate answer roles.
        order = torch.rand(batch, PAIRS, generator=self.generator).argsort(dim=-1)
        reverse = torch.rand(batch, PAIRS, generator=self.generator) < .5

        def permute(rows):
            rows = torch.gather(rows, 1, order[..., None].expand(-1, -1, 2))
            return torch.where(reverse[..., None], rows.flip(-1), rows)

        pool = torch.arange(1, VOCABULARY + 1)[None].expand(batch, -1)
        allowed = ~pool[:, :, None].eq(identities[:, None]).any(dim=-1)
        noise_pool = pool[allowed].reshape(batch, VOCABULARY - 7)
        noise_indices = torch.randint(VOCABULARY - 7, (batch, DISTRACTORS), generator=self.generator)
        noise = torch.zeros(batch, DISTRACTORS, 2, dtype=torch.long)
        noise[:, :, 0] = noise_pool.gather(1, noise_indices)
        noise = torch.where((torch.rand(batch, DISTRACTORS, generator=self.generator) < .5)[..., None], noise.flip(-1), noise)
        events = torch.cat((permute(pairs), noise), dim=1)
        result = {
            "events": events,
            "cues": identities[:, :6],
            "targets": pairs.flip(-1).reshape(batch, 6) - 1,
        }
        if counterfactual:
            result.update({
                "counterfactual_events": torch.cat((permute(swapped), noise), dim=1),
                "contrast_cues": pairs[:, :, 0],
                "contrast_targets": pairs[:, :, 1] - 1,
                "counterfactual_targets": swapped[:, :, 1] - 1,
                "correction": torch.stack((identities[:, 0], identities[:, 6]), dim=-1)[:, None],
                "corrected_targets": torch.stack((identities[:, 6], identities[:, 3], identities[:, 5]), dim=-1) - 1,
            })
        return result
