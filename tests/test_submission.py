import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from rukopys import submission
from rukopys.config import CLASS_PROMPT, PROMPTS
from rukopys.inference import write_submission
from rukopys.utils import read_json


class DummyModel:
    def eval(self):
        pass


class SubmissionTests(unittest.TestCase):
    def test_resume_abstention_policy_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "page.png"
            output_path = root / "predictions.json"
            csv_path = root / "submission.csv"

            Image.new("RGB", (40, 20)).save(image_path)

            pages = [
                {
                    "image": "page.png",
                    "image_path": str(image_path),
                    "width": 40,
                    "height": 20,
                    "regions": [{"bbox": [0, 0, 40, 20]}],
                }
            ]
            policy = {
                "unresolved_text": "empty",
                "invalid_type": "stop",
                "class_prompt": CLASS_PROMPT,
                "prompts": dict(PROMPTS),
                "max_tokens": 512,
            }
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
            model = DummyModel()

            with patch.object(
                submission,
                "generate",
                side_effect=predictions,
            ) as generate:
                result = submission.transcribe_frozen(
                    pages,
                    model,
                    None,
                    {},
                    output_path,
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
                resumed_result = submission.transcribe_frozen(
                    pages,
                    model,
                    None,
                    {},
                    output_path,
                    policy,
                )

            self.assertEqual(result, resumed_result)

            changed_policy = {**policy, "max_tokens": 256}

            with self.assertRaises(ValueError):
                submission.transcribe_frozen(
                    pages,
                    model,
                    None,
                    {},
                    output_path,
                    changed_policy,
                )

            image_path.write_bytes(b"changed")

            with self.assertRaises(ValueError):
                submission.transcribe_frozen(
                    pages,
                    model,
                    None,
                    {},
                    output_path,
                    policy,
                )

            write_submission(
                result,
                ["page.png"],
                csv_path,
                allow_truncated=True,
            )

            self.assertIn("page.png", csv_path.read_text(encoding="utf-8"))
            self.assertTrue(read_json(output_path)["complete"])

    def test_invalid_type_stops_without_guessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "page.png"
            output_path = root / "predictions.json"

            Image.new("RGB", (10, 10)).save(image_path)

            pages = [
                {
                    "image": "page.png",
                    "image_path": str(image_path),
                    "width": 10,
                    "height": 10,
                    "regions": [{"bbox": [0, 0, 10, 10]}],
                }
            ]
            policy = {
                "unresolved_text": "empty",
                "invalid_type": "stop",
                "class_prompt": CLASS_PROMPT,
                "prompts": dict(PROMPTS),
                "max_tokens": 512,
            }
            prediction = {
                "prediction": "unknown",
                "hit_token_limit": False,
            }

            with patch.object(
                submission,
                "generate",
                return_value=prediction,
            ):
                with self.assertRaises(ValueError):
                    submission.transcribe_frozen(
                        pages,
                        DummyModel(),
                        None,
                        {},
                        output_path,
                        policy,
                    )

            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
