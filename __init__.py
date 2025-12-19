"""
SGD Prompt Optimization Framework

A framework for optimizing JudgePrompt in RLAIF systems using SGD algorithm concepts.
"""

from .judge_prompt import JudgePrompt
from .forward_pass import ForwardPass
from .loss_functions import LossFunctions
from .gradient_agent import GradientAgent
from .optimizer import PromptOptimizer
from .lr_scheduler import LRScheduler
from .batch_sampler import BatchSampler
from .metrics import Metrics
from .early_stopping import EarlyStopping
from .version_control import VersionControl
from .trainer import SGDPromptTrainer

__version__ = "1.0.0"

__all__ = [
    "JudgePrompt",
    "ForwardPass",
    "LossFunctions",
    "GradientAgent",
    "PromptOptimizer",
    "LRScheduler",
    "BatchSampler",
    "Metrics",
    "EarlyStopping",
    "VersionControl",
    "SGDPromptTrainer",
]
