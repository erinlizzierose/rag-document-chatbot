import importlib.util
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import yaml

from langchain_core.documents import Document

import app

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

    def test_a_nested_list_accepts_any_one_of_its_terms(self):
        question = {"must_mention": [["prune", "pruning", "delete"]]}
        passed, _ = eval_run.grade(question, "Try pruning old checkpoints.", [])
        self.assertTrue(passed)

    def test_a_nested_list_fails_when_none_of_its_terms_appear(self):
        question = {"must_mention": [["prune", "delete"]]}
        passed, problems = eval_run.grade(question, "Restart the server.", [])
        self.assertFalse(passed)
        self.assertEqual(problems, ["missing: prune or delete"])

    def test_a_curly_apostrophe_refusal_counts(self):
        self.assertTrue(eval_run.is_refusal("I don’t know."))

    def test_isnt_covered_counts_as_a_refusal(self):
        self.assertTrue(eval_run.is_refusal("That isn't covered in the documents."))

    def test_a_caveat_on_a_correct_answer_is_not_a_decline(self):
        question = {"must_mention": ["checkpointer"]}
        answer = "Use a checkpointer. The context does not provide details on pooling."
        self.assertEqual(eval_run.grade(question, answer, []), (True, []))

    def test_a_refusal_on_an_answerable_question_still_fails(self):
        question = {"must_mention": ["checkpointer"]}
        passed, problems = eval_run.grade(question, "I do not know.", [])
        self.assertFalse(passed)
        self.assertEqual(
            problems, ["missing: checkpointer", "declined a question the documents cover"]
        )

    def test_a_term_does_not_match_inside_another_word(self):
        question = {"must_mention": ["lose"]}
        passed, _ = eval_run.grade(question, "Close the connection first.", [])
        self.assertFalse(passed)

    def test_a_term_matches_the_start_of_a_longer_word(self):
        question = {"must_mention": ["embed"]}
        passed, _ = eval_run.grade(question, "Pass an embedding function.", [])
        self.assertTrue(passed)


class QuestionSetTest(unittest.TestCase):
    def test_every_refusal_question_can_be_selected_with_the_refuse_filter(self):
        # `python evals/run.py refuse` is how you check refusals alone after a prompt change.
        questions = yaml.safe_load(eval_run.QUESTIONS_FILE.read_text())
        for question in questions:
            if question.get("should_refuse"):
                self.assertTrue(question["id"].startswith("refuse-"), question["id"])


class MainTest(unittest.TestCase):
    def run_main(self, answer_result, argv):
        output = io.StringIO()
        with mock.patch.object(eval_run, "answer_question", return_value=answer_result), \
             mock.patch.object(eval_run.sys, "argv", argv), \
             redirect_stdout(output):
            exit_code = eval_run.main()
        return exit_code, output.getvalue()

    def test_empty_index_stops_instead_of_scoring(self):
        exit_code, output = self.run_main((app.NO_INDEX_MESSAGE, [], "q"), ["run.py"])
        self.assertEqual(exit_code, 1)
        self.assertIn("index is empty", output)
        self.assertNotIn("PASS", output)

    def test_refusal_questions_with_retrieved_docs_are_still_graded(self):
        result = ("I do not know.", [doc_from("data/langchain-persistence.md")], "q")
        exit_code, output = self.run_main(result, ["run.py", "refuse"])
        self.assertEqual(exit_code, 0)
        questions = yaml.safe_load(eval_run.QUESTIONS_FILE.read_text())
        refusals = [q for q in questions if "refuse" in q["id"]]
        self.assertEqual(output.count("PASS"), len(refusals))


if __name__ == "__main__":
    unittest.main()
