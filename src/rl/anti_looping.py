"""Anti-Looping, Tag Progression, and Entropy Floor Controls (Requirement R4).

Provides:
- RollingNGramTracker: O(1) amortized n-gram recurrence tracker.
- FourGramRepetitionCriteria: Hugging Face StoppingCriteria halting at >2 occurrences.
- XMLProgressionTracker: Forward-only 7-state FSM with per-stage token limits.
- EntropyFloorController: Dual ascent entropy bonus controller for GRPO.
"""

from typing import List, Tuple, Dict, Any, Optional, Set, Union
import re
import torch
from transformers import StoppingCriteria


class RollingNGramTracker:
    """Tracks token n-gram frequencies in O(1) amortized time, optionally within a sliding window."""

    def __init__(self, n: int = 4, max_occurrences: int = 2, window_size: Optional[int] = None):
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        if max_occurrences <= 0:
            raise ValueError(f"max_occurrences must be positive, got {max_occurrences}")
        if window_size is not None and window_size < n:
            raise ValueError(f"window_size must be >= n, got {window_size}")

        self.n = n
        self.max_occurrences = max_occurrences
        self.window_size = window_size
        self.history: List[int] = []
        self.counts: Dict[Tuple[int, ...], int] = {}
        self.loop_detected: bool = False
        self.violating_ngram: Optional[Tuple[int, ...]] = None

    def reset(self) -> None:
        """Resets tracker state."""
        self.history.clear()
        self.counts.clear()
        self.loop_detected = False
        self.violating_ngram = None

    def update(self, token_id: int) -> bool:
        """Appends token_id and returns True if loop threshold is exceeded (> max_occurrences)."""
        self.history.append(int(token_id))

        # Evict n-gram that exited the sliding window if window_size is configured
        if self.window_size is not None and len(self.history) > self.window_size:
            evict_start = len(self.history) - self.window_size - 1
            evict_ngram = tuple(self.history[evict_start : evict_start + self.n])
            if evict_ngram in self.counts:
                self.counts[evict_ngram] -= 1
                if self.counts[evict_ngram] <= 0:
                    del self.counts[evict_ngram]

        if len(self.history) < self.n:
            return False

        ngram = tuple(self.history[-self.n:])
        cnt = self.counts.get(ngram, 0) + 1
        self.counts[ngram] = cnt

        if cnt > self.max_occurrences:
            self.loop_detected = True
            self.violating_ngram = ngram
            return True

        return False

    def get_max_occurrence(self) -> int:
        """Returns the maximum count of any n-gram seen so far."""
        if not self.counts:
            return 0
        return max(self.counts.values())


class FourGramRepetitionCriteria(StoppingCriteria):
    """Hugging Face StoppingCriteria halting rollout when a 4-gram repeats > 2 times.

    Assigns repetition penalty r_rep = -0.5 when triggered.
    """

    def __init__(
        self,
        max_occurrences: int = 2,
        penalty: float = -0.5,
        n: int = 4,
        prompt_lengths: Optional[List[int]] = None,
        stop_on_any: bool = True,
        window_size: Optional[int] = None,
    ):
        super().__init__()
        self.n = n
        self.max_occurrences = max_occurrences
        self.penalty = penalty
        self.prompt_lengths = list(prompt_lengths) if prompt_lengths is not None else []
        self.stop_on_any = stop_on_any
        self.window_size = window_size

        self.trackers: List[RollingNGramTracker] = []
        self.has_looped: List[bool] = []
        self.repetition_penalties: List[float] = []
        self.processed_tokens: List[int] = []

    def reset(self, batch_size: int, prompt_lengths: Optional[List[int]] = None) -> None:
        """Re-initializes state for a new batch."""
        if prompt_lengths is not None:
            self.prompt_lengths = list(prompt_lengths)
        else:
            self.prompt_lengths = [0] * batch_size

        while len(self.prompt_lengths) < batch_size:
            self.prompt_lengths.append(0)

        self.trackers = [
            RollingNGramTracker(n=self.n, max_occurrences=self.max_occurrences, window_size=self.window_size)
            for _ in range(batch_size)
        ]
        self.has_looped = [False] * batch_size
        self.repetition_penalties = [0.0] * batch_size
        self.processed_tokens = [0] * batch_size

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> bool:
        """Evaluates batch generation step against 4-gram repetition."""
        batch_size = input_ids.shape[0]
        if len(self.trackers) != batch_size:
            self.reset(batch_size, self.prompt_lengths)

        seq_len = input_ids.shape[1]
        for b in range(batch_size):
            p_len = self.prompt_lengths[b] if b < len(self.prompt_lengths) else 0
            if seq_len <= p_len:
                continue

            start_idx = max(p_len, self.processed_tokens[b])
            for idx in range(start_idx, seq_len):
                tok = input_ids[b, idx].item()
                is_loop = self.trackers[b].update(tok)
                if is_loop:
                    self.has_looped[b] = True
                    self.repetition_penalties[b] = self.penalty
            self.processed_tokens[b] = seq_len

        if self.stop_on_any:
            return any(self.has_looped)
        return all(self.has_looped)

    def is_loop(self, batch_idx: int = 0) -> bool:
        """Returns True if the sequence at batch_idx looped."""
        if 0 <= batch_idx < len(self.has_looped):
            return self.has_looped[batch_idx]
        return False

    def get_penalty(self, batch_idx: int = 0) -> float:
        """Returns repetition penalty for the sequence at batch_idx."""
        if 0 <= batch_idx < len(self.repetition_penalties):
            return self.repetition_penalties[batch_idx]
        return 0.0

    def evaluate_sequence(
        self,
        token_ids: Union[List[int], torch.Tensor]
    ) -> Tuple[bool, float, int]:
        """Evaluates a completed sequence directly.

        Returns:
            (has_looped, penalty, max_occurrences)
        """
        if isinstance(token_ids, torch.Tensor):
            tokens = token_ids.view(-1).tolist()
        else:
            tokens = list(token_ids)

        tracker = RollingNGramTracker(n=self.n, max_occurrences=self.max_occurrences)
        looped = False
        for tok in tokens:
            if tracker.update(tok):
                looped = True
                break

        penalty = self.penalty if looped else 0.0
        return looped, penalty, tracker.get_max_occurrence()


# Backward compatibility alias
RepetitionStoppingCriteria = FourGramRepetitionCriteria


class XMLProgressionTracker:
    """Forward-only 7-state FSM enforcing stage sequence and token limits."""

    STATE_PROMPT = "PROMPT"
    STATE_EXPLORE = "EXPLORE"
    STATE_CONJECTURE = "CONJECTURE"
    STATE_TEST_EDGE_CASES = "TEST_EDGE_CASES"
    STATE_LEMMA_ISOLATE = "LEMMA_ISOLATE"
    STATE_FORMAL_PROOF = "FORMAL_PROOF"
    STATE_TERMINAL = "TERMINAL"

    STATES = [
        STATE_PROMPT,
        STATE_EXPLORE,
        STATE_CONJECTURE,
        STATE_TEST_EDGE_CASES,
        STATE_LEMMA_ISOLATE,
        STATE_FORMAL_PROOF,
        STATE_TERMINAL,
    ]

    DEFAULT_TAG_ORDER = [
        "explore",
        "conjecture",
        "test_edge_cases",
        "lemma_isolate",
        "formal_proof"
    ]

    DEFAULT_TOKEN_LIMITS = {
        "explore": 1024,
        "conjecture": 512,
        "test_edge_cases": 512,
        "lemma_isolate": 768,
        "formal_proof": 2048,
    }

    TAG_TO_STATE = {
        "explore": STATE_EXPLORE,
        "conjecture": STATE_CONJECTURE,
        "test_edge_cases": STATE_TEST_EDGE_CASES,
        "lemma_isolate": STATE_LEMMA_ISOLATE,
        "formal_proof": STATE_FORMAL_PROOF,
    }

    def __init__(
        self,
        tag_order: Optional[List[str]] = None,
        per_stage_token_limits: Optional[Dict[str, int]] = None
    ):
        self.tag_order = list(tag_order) if tag_order is not None else list(self.DEFAULT_TAG_ORDER)
        self.stage_token_limits = (
            dict(per_stage_token_limits)
            if per_stage_token_limits is not None
            else dict(self.DEFAULT_TOKEN_LIMITS)
        )

        self.current_state = self.STATE_PROMPT
        self.visited_states = [self.STATE_PROMPT]
        self.current_open_tag: Optional[str] = None
        self.completed_tags: Set[str] = set()
        self.stage_token_counts: Dict[str, int] = {tag: 0 for tag in self.tag_order}
        self.violations: List[str] = []
        self._raw_buffer: str = ""

    def reset(self) -> None:
        """Resets tracker to initial state."""
        self.current_state = self.STATE_PROMPT
        self.visited_states = [self.STATE_PROMPT]
        self.current_open_tag = None
        self.completed_tags.clear()
        self.stage_token_counts = {tag: 0 for tag in self.tag_order}
        self.violations.clear()
        self._raw_buffer = ""

    def _get_expected_next_tag(self) -> Optional[str]:
        """Returns the next expected tag in sequence."""
        for tag in self.tag_order:
            if tag not in self.completed_tags and tag != self.current_open_tag:
                return tag
        return None

    def feed_token(self, token_str: str) -> Tuple[bool, Optional[str]]:
        """Processes an incoming token string.

        Returns (is_valid, violation_reason).
        """
        self._raw_buffer += token_str

        # Accumulate token count if inside an active tag block
        if self.current_open_tag:
            self.stage_token_counts[self.current_open_tag] += 1
            limit = self.stage_token_limits.get(self.current_open_tag, float("inf"))
            if self.stage_token_counts[self.current_open_tag] > limit:
                violation = (
                    f"Token budget exceeded in <{self.current_open_tag}>: "
                    f"{self.stage_token_counts[self.current_open_tag]} > {limit}"
                )
                if violation not in self.violations:
                    self.violations.append(violation)
                return False, violation

        # Check for newly completed opening tags
        for tag in self.tag_order:
            open_marker = f"<{tag}>"
            if open_marker in self._raw_buffer:
                self._raw_buffer = self._raw_buffer.replace(open_marker, "")

                if self.current_open_tag is not None:
                    violation = f"Nesting violation: opened <{tag}> inside <{self.current_open_tag}>"
                    self.violations.append(violation)
                    return False, violation

                if tag in self.completed_tags:
                    violation = f"Backtracking violation: tag <{tag}> reopened after completion"
                    self.violations.append(violation)
                    return False, violation

                expected_tag = self._get_expected_next_tag()
                if tag != expected_tag:
                    violation = f"Stage skipping violation: expected <{expected_tag}> but received <{tag}>"
                    self.violations.append(violation)
                    return False, violation

                self.current_open_tag = tag
                next_state = self.TAG_TO_STATE.get(tag, self.current_state)
                self.current_state = next_state
                self.visited_states.append(next_state)

            close_marker = f"</{tag}>"
            if close_marker in self._raw_buffer:
                self._raw_buffer = self._raw_buffer.replace(close_marker, "")

                if self.current_open_tag != tag:
                    violation = f"Mismatched closing tag </{tag}> when open tag was {self.current_open_tag}"
                    self.violations.append(violation)
                    return False, violation

                self.completed_tags.add(tag)
                self.current_open_tag = None

                if len(self.completed_tags) == len(self.tag_order):
                    self.current_state = self.STATE_TERMINAL
                    self.visited_states.append(self.STATE_TERMINAL)

        return len(self.violations) == 0, (self.violations[-1] if self.violations else None)

    def validate_trace(
        self,
        text: str,
        stage_token_counts: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """Validates a complete text trace against FSM progression rules.

        Returns:
            Dict containing valid flag, current state, visited states, violations, and counts.
        """
        violations: List[str] = []
        tag_positions: Dict[str, Tuple[int, int]] = {}

        # 1. Check tag presence and counts
        for tag in self.tag_order:
            open_count = text.count(f"<{tag}>")
            close_count = text.count(f"</{tag}>")
            if open_count == 0 or close_count == 0:
                violations.append(f"Missing required tag <{tag}> (open={open_count}, close={close_count})")
            elif open_count > 1 or close_count > 1:
                violations.append(f"Duplicate tag <{tag}> (open={open_count}, close={close_count})")
            else:
                o_pos = text.find(f"<{tag}>")
                c_pos = text.find(f"</{tag}>")
                if o_pos >= c_pos:
                    violations.append(f"Malformed tag order for <{tag}>: close before open")
                tag_positions[tag] = (o_pos, c_pos)

        # 2. Strict sequential order and no nesting / backtracking
        last_close = -1
        for i, tag in enumerate(self.tag_order):
            if tag not in tag_positions:
                continue
            o_pos, c_pos = tag_positions[tag]

            if o_pos < last_close:
                violations.append(f"Backtracking detected: <{tag}> opened before previous stage closed")

            # Check inside content for nested tags
            content = text[o_pos + len(f"<{tag}>"):c_pos]
            for other_tag in self.tag_order:
                if f"<{other_tag}>" in content or f"</{other_tag}>" in content:
                    violations.append(f"Nesting detected inside <{tag}> with <{other_tag}>")

            last_close = c_pos + len(f"</{tag}>")

        # 3. Check for stage skipping
        present_tags = [t for t in self.tag_order if t in tag_positions]
        if present_tags != self.tag_order:
            missing = [t for t in self.tag_order if t not in tag_positions]
            if not any("Missing required tag" in v for v in violations):
                violations.append(f"Stage skipped: missing {missing}")

        # 4. Token limit checks
        computed_counts: Dict[str, int] = {}
        for tag in self.tag_order:
            if tag in tag_positions:
                o_pos, c_pos = tag_positions[tag]
                content = text[o_pos + len(f"<{tag}>"):c_pos]
                if stage_token_counts and tag in stage_token_counts:
                    count = stage_token_counts[tag]
                else:
                    # Token estimate by whitespace and punctuation
                    count = len(re.findall(r"\w+|[^\w\s]", content))
                computed_counts[tag] = count
                limit = self.stage_token_limits.get(tag, float("inf"))
                if count > limit:
                    violations.append(f"Token budget exceeded in <{tag}>: {count} > {limit}")
            else:
                computed_counts[tag] = 0

        is_valid = (len(violations) == 0)
        final_state = self.STATE_TERMINAL if is_valid else self.STATE_PROMPT

        return {
            "valid": is_valid,
            "current_state": final_state,
            "visited_states": self.STATES if is_valid else [self.STATE_PROMPT],
            "violations": violations,
            "stage_token_counts": computed_counts,
            "error_message": violations[0] if violations else None
        }


# Backward compatibility alias
XMLTagProgressionFSM = XMLProgressionTracker


class EntropyFloorController:
    """Dual ascent controller regulating policy entropy bonus for GRPO training.

    Maintains output entropy at or above floor_nats to prevent collapse
    into safe-phrase loops.
    """

    def __init__(
        self,
        floor_nats: float = 0.25,
        lr: float = 0.01,
        beta_min: float = 0.001,
        beta_max: float = 0.10,
        beta_init: float = 0.01
    ):
        if floor_nats < 0:
            raise ValueError(f"floor_nats must be non-negative, got {floor_nats}")
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if beta_min > beta_max:
            raise ValueError(f"beta_min ({beta_min}) must not exceed beta_max ({beta_max})")

        self.floor_nats = floor_nats
        self.lr = lr
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta = max(beta_min, min(beta_max, beta_init))
        self.history: List[Dict[str, float]] = []

    def compute_entropy(
        self,
        logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes mean Shannon entropy in nats over active completion tokens."""
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        token_entropy = -(probs * log_probs).sum(dim=-1)

        if mask is not None:
            mask_f = mask.to(dtype=logits.dtype)
            num_tokens = mask_f.sum().clamp(min=1.0)
            mean_entropy = (token_entropy * mask_f).sum() / num_tokens
        else:
            mean_entropy = token_entropy.mean()

        return mean_entropy

    def update(self, current_entropy: Union[float, torch.Tensor]) -> float:
        """Updates beta via dual ascent based on distance to floor.

        Returns updated beta value.
        """
        if isinstance(current_entropy, torch.Tensor):
            ent_val = float(current_entropy.item())
        else:
            ent_val = float(current_entropy)

        delta = self.floor_nats - ent_val
        updated_beta = self.beta + self.lr * delta
        self.beta = float(max(self.beta_min, min(self.beta_max, updated_beta)))

        self.history.append({
            "entropy": ent_val,
            "delta": delta,
            "beta": self.beta
        })
        return self.beta

    def update_beta(self, current_entropy: Union[float, torch.Tensor]) -> float:
        """Alias for update."""
        return self.update(current_entropy)

    def loss_penalty(self, entropy: torch.Tensor) -> torch.Tensor:
        """Calculates loss term (-beta * H) to encourage entropy."""
        return -self.beta * entropy


# Backward compatibility alias
EntropyFloorRegularizer = EntropyFloorController
