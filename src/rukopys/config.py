"""Shared model settings, region types, and prompts."""

TYPES = (
    "handwritten",
    "printed",
    "formula",
    "table",
    "annotation",
    "image",
    "graph",
)


MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


TEXT_PROMPT = (
    "Transcribe all text in this cropped Ukrainian document region in reading order. "
    "Preserve original spelling, punctuation, historical spelling and incomplete words. "
    "Do not correct, translate, explain or complete the text. "
    "Return only the transcription."
)


PROMPTS = {
    "handwritten": TEXT_PROMPT,
    "printed": TEXT_PROMPT,
    "annotation": TEXT_PROMPT,
    "formula": (
        "Transcribe every mathematical expression in this crop as LaTeX. "
        "Preserve symbols and structure. "
        "Return only the transcription, without Markdown fences or explanations."
    ),
    "table": (
        "Transcribe this table exactly. "
        "Separate cells with | and rows with newlines. "
        "Preserve empty cells and original spelling. "
        "Return only the table, without explanations or Markdown separator rows."
    ),
}


CLASS_PROMPT = (
    "Classify the main content of this document crop. "
    "Return exactly one label: "
    "handwritten, printed, formula, table, annotation, image, graph. "
    "Use annotation for marginal notes, signatures or correction marks; "
    "formula for mathematical expressions; "
    "table for tabular content; "
    "graph for plotted charts; "
    "image for non-text illustrations. "
    "No explanation."
)
