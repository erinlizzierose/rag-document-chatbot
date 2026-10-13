import importlib.util
import unittest
from pathlib import Path

from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("eval_run", ROOT / "evals" / "run.py")
eval_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_run)


def doc_from(source):
    return Document("chunk text", metadata={"source": source})


class GradeTest(unittest.TestCase):
    def test_refusal_question_passes_on_a_refusal(self):
        question = {"should_refuse": True}
        self.assertEqual(eval_run.grade(question, "I do not know.", []), (True, []))

    def test_refusal_question_fails_on_an_answer(self):
        question = {"should_refuse": True}
        passed, problems = eval_run.grade(question, "Paris.", [])
        self.assertFalse(passed)
        self.assertEqual(problems, ["answered a question it should have declined"])

    def test_must_mention_is_case_insensitive(self):
        question = {"must_mention": ["Checkpointer"], "expect_source": "data/a.md"}
        passed, _ = eval_run.grade(question, "Use a checkpointer.", [doc_from("data/a.md")])
        self.assertTrue(passed)

    def test_any_listed_source_counts(self):
        question = {"must_mention": [], "expect_source": ["data/a.md", "data/b.md"]}
        passed, _ = eval_run.grade(question, "Answer.", [doc_from("data/b.md")])
        self.assertTrue(passed)

    def test_missing_source_is_reported(self):
        question = {"must_mention": [], "expect_source": "data/a.md"}
        passed, problems = eval_run.grade(question, "Answer.", [doc_from("data/z.md")])
        self.assertFalse(passed)
        self.assertEqual(problems, ["never retrieved data/a.md"])


if __name__ == "__main__":
    unittest.main()
