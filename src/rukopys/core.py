"""Compatibility exports for the original core module."""

from .config import TYPES, MODEL_ID, TEXT_PROMPT, PROMPTS, CLASS_PROMPT
from .utils import read_json, atomic_json, digest, file_digest, valid_bbox
from .metrics import edit_distance, cer_summary
from .data import make_split, prepare_data
from .models import load_qwen, conversation, generate, parse_type
from .evaluate import evaluate
from .train import training_examples, supervised_batch, train
from .inference import detect_pages, transcribe_pages, write_submission

__all__ = ['TYPES', 'MODEL_ID', 'TEXT_PROMPT', 'PROMPTS', 'CLASS_PROMPT', 'read_json', 'atomic_json', 'digest', 'file_digest', 'valid_bbox', 'edit_distance', 'cer_summary', 'make_split', 'prepare_data', 'load_qwen', 'conversation', 'generate', 'parse_type', 'evaluate', 'training_examples', 'supervised_batch', 'train', 'detect_pages', 'transcribe_pages', 'write_submission']
