"""Reinforcement learning post-training components for Chalk pipeline."""

# Requirement R4: Anti-Looping and Attractor Prevention
from src.rl.anti_looping import (
    RollingNGramTracker,
    FourGramRepetitionCriteria,
    RepetitionStoppingCriteria,
    XMLProgressionTracker,
    XMLTagProgressionFSM,
    EntropyFloorController,
    EntropyFloorRegularizer,
)

# Requirement R5: Rottweiler Verification and Format Discrimination
from src.rl.rottweiler_verifier import (
    XMLScaffoldValidator,
    RottweilerSymPyVerifier,
    RottweilerVerifier,
    extract_boxed_answer,
    clean_latex_math,
    is_safe_math_expression,
    TAG_NAMES,
)
from src.rl.format_replay_buffer import (
    PromptType,
    FormatDiscriminationReplayBuffer,
    FormatDiscriminationRewardEngine,
)

# Requirements R2 and R3 (conditional imports when present)
try:
    from src.rl.modified_grpo_loss import ModifiedGRPOLoss, GRPOLossOutput
except ImportError:
    ModifiedGRPOLoss = None
    GRPOLossOutput = None

try:
    from src.rl.calibrated_abstention import CalibratedAbstentionRewardEngine
except ImportError:
    CalibratedAbstentionRewardEngine = None

__all__ = [
    "RollingNGramTracker",
    "FourGramRepetitionCriteria",
    "RepetitionStoppingCriteria",
    "XMLProgressionTracker",
    "XMLTagProgressionFSM",
    "EntropyFloorController",
    "EntropyFloorRegularizer",
    "XMLScaffoldValidator",
    "RottweilerSymPyVerifier",
    "RottweilerVerifier",
    "extract_boxed_answer",
    "clean_latex_math",
    "is_safe_math_expression",
    "TAG_NAMES",
    "PromptType",
    "FormatDiscriminationReplayBuffer",
    "FormatDiscriminationRewardEngine",
    "ModifiedGRPOLoss",
    "GRPOLossOutput",
    "CalibratedAbstentionRewardEngine",
]
