import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rukopys import submission
from rukopys.config import CLASS_PROMPT, PROMPTS
from rukopys.inference import write_submission
from rukopys.utils import read_json


class SubmissionTests(unittest.TestCase):
    """Check frozen inference, checkpoint reuse, and submission export."""

    def test_resume_abstention_policy_and_csv(self):
        """Verify abstention, result reuse, configuration checks, and CSV export."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "page.png"
            Image.new("RGB", (40, 20)).save(image)

            pages = [
                {
                    "image": "page.png",
                    "image_path": str(image),
                    "width": 40,
                    "height": 20,
                    "regions": [
                        {"bbox": [0, 0, 40, 20]},
                    ],
                }
            ]
            policy = {
                "unresolved_text": "empty",
                "invalid_type": "stop",
                "class_prompt": CLASS_PROMPT,
                "prompts": dict(PROMPTS),
                "max_tokens": 512,
            }

            class Model:
                """Provide the minimal model interface required by inference."""

                def eval(self):
                    """Provide a no-op evaluation mode for the test."""
                    pass

            predictions = [
                {
                    "prediction": "handwritten",
                    "hit_token_limit": False,
                },
                {
                    "prediction": "bad repeated text",
                    "hit_token_limit": True,
                },
            ]
            output = root / "predictions.json"

            with patch.object(
                submission,
                "generate",
                side_effect=predictions,
            ) as generate:
                result = submission.transcribe_frozen(
                    pages,
                    Model(),
                    None,
                    {},
                    output,
                    policy,
                )

                self.assertEqual(generate.call_count, 2)
                self.assertEqual(result[0]["regions"][0]["text"], "")
                self.assertTrue(result[0]["diagnostics"][0]["transcription"]["hit_token_limit"])

            with patch.object(
                submission,
                "generate",
                side_effect=AssertionError("No regeneration"),
            ):
                again = submission.transcribe_frozen(
                    pages,
                    Model(),
                    None,
                    {},
                    output,
                    policy,
                )

            self.assertEqual(result, again)

            changed = {**policy, "max_tokens": 256}

            with self.assertRaises(ValueError):
                submission.transcribe_frozen(
                    pages,
                    Model(),
                    None,
                    {},
                    output,
                    changed,
                )

            image.write_bytes(b"changed")

            with self.assertRaises(ValueError):
                submission.transcribe_frozen(
                    pages,
                    Model(),
                    None,
                    {},
                    output,
                    policy,
                )

            csv_path = root / "submission.csv"
            write_submission(
                result,
                ["page.png"],
                csv_path,
                allow_truncated=True,
            )

            self.assertIn("page.png", csv_path.read_text())
            self.assertTrue(read_json(output)["complete"])

    def test_invalid_type_stops_without_guessing(self):
        """Verify that an invalid type stops inference without saving predictions."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "p.png"
            Image.new("RGB", (10, 10)).save(image)

            pages = [
                {
                    "image": "p.png",
                    "image_path": str(image),
                    "regions": [
                        {"bbox": [0, 0, 10, 10]},
                    ],
                }
            ]

            class Model:
                """Provide the minimal model interface required by inference."""

                def eval(self):
                    """Provide a no-op evaluation mode for the test."""
                    pass

            policy = {
                "unresolved_text": "empty",
                "invalid_type": "stop",
                "class_prompt": CLASS_PROMPT,
                "prompts": dict(PROMPTS),
                "max_tokens": 512,
            }

            with patch.object(
                submission,
                "generate",
                return_value={
                    "prediction": "unknown",
                    "hit_token_limit": False,
                },
            ):
                with self.assertRaises(ValueError):
                    submission.transcribe_frozen(
                        pages,
                        Model(),
                        None,
                        {},
                        root / "out.json",
                        policy,
                    )

            self.assertFalse((root / "out.json").exists())


if __name__ == "__main__":
    unittest.main()
