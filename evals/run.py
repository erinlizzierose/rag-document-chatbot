"""Run the eval set against the chatbot and print a score.

    python evals/run.py              # every question
    python evals/run.py trim         # only questions whose id contains "trim"

Imports app.py directly rather than driving the UI. That is why load_dotenv() sits at module
scope in app.py and why answer_question() returns the retrieved documents alongside the answer:
this script needs both, and neither is reachable through Streamlit.

Each question costs one embedding call and one chat call. History is not passed, so the rewrite
call never fires. The full set runs for under a cent.
"""

import logging
import re
import sys
from pathlib import Path

import yaml

# app.py is one directory up. Imported before anything touches it so the path is set first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Importing app.py imports Streamlit, which can log warnings when used outside `streamlit run`.
# Nothing is wrong when it does; the warnings are just noise that would bury the results.
logging.getLogger("streamlit").setLevel(logging.ERROR)

from app import answer_question, describe_source  # noqa: E402


QUESTIONS_FILE = Path(__file__).parent / "questions.yaml"

# Grading a refusal means looking for one of these in the answer. It is a crude test and it is
# meant to be: the alternative is a second model call to judge the first, which costs money,
# introduces its own failures, and is not worth it until this proves inadequate. If a genuine
# refusal is ever scored as a miss, add its phrasing here rather than reaching for a judge.
REFUSAL_MARKERS = (
    "do not know",
    "don't know",
    "not in the context",
    "not contain",
    "doesn't contain",
    "no information",
    "does not provide",
    "doesn't provide",
    "not provided",
    "not covered",
    "isn't covered",
    "does not mention",
    "doesn't mention",
    "cannot answer",
    "can't answer",
    "could not find",
    "couldn't find",
)


def is_refusal(answer):
    # Models write apostrophes as either ' or the curly ’, and "don’t know" missed every marker.
    lowered = answer.lower().replace("’", "'")
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def mentions(answer, term):
    """True if term appears at the start of a word in answer, ignoring case.

    Anchored at the start of a word only. Plain substring matching let "lose" pass on "close";
    anchoring both ends would fail "embed" on "embedding" and "trim" on "trimming", which are
    exactly the variations a correct answer uses.
    """
    return re.search(r"\b" + re.escape(term), answer, re.IGNORECASE) is not None


def grade(question, answer, docs):
    """Return (passed, list of reasons it failed).

    Answer text and retrieved sources are graded separately. Retrieval usually degrades before
    answers visibly do, so a question that passes its text check while missing its expected
    source is the early warning worth seeing.
    """
    problems = []

    if question.get("should_refuse"):
        if not is_refusal(answer):
            problems.append("answered a question it should have declined")
        return not problems, problems

    # Each entry is one term, or a list of alternatives where any one counts. The list form is
    # for an answer the documents phrase more than one way: requiring one exact word fails a
    # correct answer that paraphrases, the same way requiring one exact source file would.
    missing = []
    for term in question.get("must_mention", []):
        options = term if isinstance(term, list) else [term]
        if not any(mentions(answer, option) for option in options):
            missing.append(" or ".join(options))
    if missing:
        problems.append("missing: " + ", ".join(missing))
        # Only reported alongside missing terms. Correct answers often add a caveat about what
        # the context leaves out ("the context does not provide details on pooling"), and the
        # marker list cannot tell that apart from a refusal. An answer with every required term
        # in it answered the question, whatever else it says.
        if is_refusal(answer):
            problems.append("declined a question the documents cover")

    # One string, or a list when more than one document genuinely answers the question. Any
    # one of them counts: requiring a specific file where two are equally correct fails good
    # answers, and a suite with bogus failures in it stops getting trusted.
    expected = question.get("expect_source") or []
    if isinstance(expected, str):
        expected = [expected]
    if expected:
        sources = {doc.metadata.get("source") for doc in docs}
        if not sources.intersection(expected):
            problems.append("never retrieved " + " or ".join(expected))

    return not problems, problems


def main():
    questions = yaml.safe_load(QUESTIONS_FILE.read_text())

    if len(sys.argv) > 1:
        needle = sys.argv[1].lower()
        questions = [q for q in questions if needle in q["id"].lower()]
        if not questions:
            print(f"No question ids contain {sys.argv[1]!r}")
            return 1

    failures = []

    for question in questions:
        answer, docs, _ = answer_question(question["question"])

        # An empty index turns every answer into the "run ingest.py" message, which contains a
        # refusal marker, so the refusal questions would pass. Stop rather than print a score
        # that means nothing.
        if not docs:
            print("The index is empty. Run `python ingest.py` first, then try again.")
            return 1

        passed, problems = grade(question, answer, docs)

        mark = "PASS" if passed else "FAIL"
        print(f"{mark}  {question['id']}")

        if not passed:
            failures.append(question["id"])
            print(f"      Q: {question['question']}")
            print(f"      A: {answer.strip()[:300]}")
            for problem in problems:
                print(f"      ! {problem}")
            # Retrieval is the usual culprit, so show what came back on every failure.
            print(f"      retrieved: {', '.join(describe_source(d) for d in docs)}")
            print()

    total = len(questions)
    print(f"\n{total - len(failures)}/{total} passed")
    if failures:
        print("failed: " + ", ".join(failures))

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
