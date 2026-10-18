"""Compatibility exports for the original core module."""

from .config import CLASS_PROMPT, MODEL_ID, PROMPTS, TEXT_PROMPT, TYPES
from .data import make_split, prepare_data
from .evaluate import evaluate
from .inference import detect_pages, transcribe_pages, write_submission
from .metrics import cer_summary, edit_distance
from .models import conversation, generate, load_qwen, parse_type
from .train import supervised_batch, train, training_examples
from .utils import atomic_json, digest, file_digest, read_json, valid_bbox

__all__ = [
    "TYPES",
    "MODEL_ID",
    "TEXT_PROMPT",
    "PROMPTS",
    "CLASS_PROMPT",
    "read_json",
    "atomic_json",
    "digest",
    "file_digest",
    "valid_bbox",
    "edit_distance",
    "cer_summary",
    "make_split",
    "prepare_data",
    "load_qwen",
    "conversation",
    "generate",
    "parse_type",
    "evaluate",
    "training_examples",
    "supervised_batch",
    "train",
    "detect_pages",
    "transcribe_pages",
    "write_submission",
]
