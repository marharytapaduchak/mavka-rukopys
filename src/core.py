"""Compatibility exports for legacy notebooks and scripts."""

from rukopys.core import CLASS_PROMPT as CLASS_PROMPT
from rukopys.core import MODEL_ID as MODEL_ID
from rukopys.core import PROMPTS as PROMPTS
from rukopys.core import TEXT_PROMPT as TEXT_PROMPT
from rukopys.core import TYPES as TYPES
from rukopys.core import atomic_json as atomic_json
from rukopys.core import cer_summary as cer_summary
from rukopys.core import conversation as conversation
from rukopys.core import detect_pages as detect_pages
from rukopys.core import digest as digest
from rukopys.core import edit_distance as edit_distance
from rukopys.core import evaluate as evaluate
from rukopys.core import file_digest as file_digest
from rukopys.core import generate as generate
from rukopys.core import load_qwen as load_qwen
from rukopys.core import make_split as make_split
from rukopys.core import parse_type as parse_type
from rukopys.core import prepare_data as prepare_data
from rukopys.core import read_json as read_json
from rukopys.core import supervised_batch as supervised_batch
from rukopys.core import train as train
from rukopys.core import training_examples as training_examples
from rukopys.core import transcribe_pages as transcribe_pages
from rukopys.core import valid_bbox as valid_bbox
from rukopys.core import write_submission as write_submission

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
