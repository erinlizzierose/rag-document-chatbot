# Phase 3: Local Model Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the same RAG pipeline with OpenAI and with a local Ollama model, with the embedding model and the chat model as separate settings, and measure all four combinations with the eval set.

**Architecture:** A new `models.py` becomes the one place that knows which embedding model and chat model are in use. It reads two settings, `EMBEDDING_BACKEND` and `CHAT_BACKEND` (each `openai` or `ollama`). `ingest.py`, `app.py` and `evals/run.py` ask it for models instead of building them. Each embedding model gets its own Chroma collection, so the two indexes sit side by side and switching backends can never compare a question against vectors from a different model. Before the comparison, the eval gets fairer grading and three "near-miss" refusal questions, so it can measure the weakness a small local model is expected to have.

**Tech Stack:** Python 3.13 venv (3.10+ supported), LangChain 1.x split packages, `langchain-ollama` 1.1.0, Chroma, Streamlit, Ollama 0.40.x, `unittest`.

**Spec:** `ROADMAP.md`, section "Phase 3: Run it locally too, and compare", plus the first two entries under "Known issues". There is no separate design doc. The decisions below were researched on 2026-10-18 and are recorded in Global Constraints.

## Global Constraints

- Work in the worktree `/Users/erinrose/.superset/worktrees/chatbot/quaint-biology-92a8aeba` on branch `repo-architecture-guide-92a8aeba`. Never commit on `main`.
- Shell variables used throughout:
  ```bash
  W=/Users/erinrose/.superset/worktrees/chatbot/quaint-biology-92a8aeba
  MAIN=/Users/erinrose/projects/learning/chatbot
  PY=$MAIN/.venv/bin/python
  ```
- Offline unit tests run from the worktree: `cd $W && $PY -m unittest discover -s tests -v`. They must never need network, an API key, or a running Ollama.
- Anything that reads `data/`, `vectorstore/` or the API key runs from the main checkout with the worktree's code: `cd $MAIN && ( set -a; . ./.env; set +a; $PY $W/evals/run.py )`. Never print `.env` or the key.
- Paid OpenAI calls only in the steps that say so. Budget for the whole plan: under 10 cents.
- Pins: `langchain-core==1.4.8` must not change. Add `langchain-ollama==1.1.0` (PyPI, 2026-04-07, requires `langchain-core>=1.2.21,<2` and `ollama>=0.6.1,<1`).
- Models, exactly: OpenAI embeddings `text-embedding-3-small`, OpenAI chat `gpt-4o-mini` at `temperature=0`, Ollama embeddings `nomic-embed-text`, Ollama chat `llama3.2:3b` at `temperature=0` with `num_ctx=8192`.
- `nomic-embed-text` requires task prefixes (its model card: the text "must include a task instruction prefix"): `search_document: ` before stored text, `search_query: ` before questions. `OllamaEmbeddings` does not add them.
- Ollama defaults to a 4096-token context window on machines with under 24 GiB of VRAM (docs.ollama.com/context-length). This Mac is an Apple M2 with 8 GB of RAM.
- Keep `CHUNK_SIZE = 750`, `CHUNK_OVERLAP = 150`, `RETRIEVAL_K = 6`, and both prompts identical across backends. The comparison varies only the models.
- Writing: no em dashes, no emojis, plain English, comments match the surrounding density and explain why.
- Commits: one sentence, no `Co-Authored-By` trailer. The repo's `user.email` is already the GitHub noreply address. Never push. Never use bare `git stash`.

## Review Focus

1. **A prompt longer than Ollama's context window.** Expected: never happens, because `num_ctx` is set explicitly. Pinned by `test_ollama_chat_model_sets_the_context_window` (Task 3) and measured against a long conversation in Task 6 Step 2.
2. **Switching `EMBEDDING_BACKEND` without re-running `ingest.py`.** Expected: the app says to run `ingest.py`, not wrong answers. Pinned by `test_each_embedding_backend_gets_its_own_collection` (Task 3), `test_each_embedding_model_writes_its_own_collection` (Task 4) and `test_retriever_reads_the_collection_for_the_current_embedding_model` (Task 4).
3. **Ollama not running, or a model not pulled.** Expected: one readable sentence in the app, the eval and ingest, not a traceback. Pinned by `test_ollama_not_running_is_reported` (Task 3), `test_a_setup_problem_stops_before_touching_the_index` and `test_a_setup_problem_stops_before_any_question` (Task 4).
4. **A typo or odd capitalisation in a backend setting** (`Ollama`, `olama`). Expected: case and spaces ignored, unknown values reported clearly. Pinned by `test_is_case_insensitive_and_ignores_spaces` and `test_unknown_backend_is_reported_not_raised` (Task 3).
5. **Running fully local with no OpenAI key.** Expected: works and never asks for a key. Pinned by `test_no_key_needed_when_everything_runs_locally` (Task 3).

---

### Task 1: Make the eval grade fairly before it compares anything

The refusal check misses refusals phrased as "isn't covered" or with a curly apostrophe, and counts a correct answer with a caveat as a refusal. Term matching is by substring, so "lose" passes on "close". A local model phrases things differently from `gpt-4o-mini`, so these would distort the comparison.

**Files:**
- Modify: `evals/run.py` (`REFUSAL_MARKERS`, `is_refusal`, `grade`, new `mentions`)
- Modify: `evals/questions.yaml` (header comment, `trim-vs-summarize`, `long-term-memory-store`)
- Test: `tests/test_eval_runner.py`

**Interfaces:**
- Produces: `mentions(answer: str, term: str) -> bool` in `evals/run.py`. `grade()` keeps its signature and return shape `(passed: bool, problems: list[str])`.

- [ ] **Step 1: Write the failing tests**

Add to `GradeTest` in `tests/test_eval_runner.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd $W && $PY -m unittest tests.test_eval_runner -v`
Expected: 4 FAIL: `test_a_curly_apostrophe_refusal_counts`, `test_isnt_covered_counts_as_a_refusal`, `test_a_caveat_on_a_correct_answer_is_not_a_decline`, `test_a_term_does_not_match_inside_another_word`. The other two new tests already pass and stay as guards.

- [ ] **Step 3: Implement**

In `evals/run.py`, add `import re` to the imports (alphabetical: after `import logging`). Replace `REFUSAL_MARKERS` and `is_refusal` with:

```python
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
```

In `grade()`, replace the block from `missing = []` down to and including the separate `if is_refusal(answer):` check with:

```python
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
```

Keep the existing any-of comment above the loop.

- [ ] **Step 4: Fix the two weak terms in `evals/questions.yaml`**

In the header, change the `must_mention` description to:

```yaml
#   must_mention  terms that all have to appear in the answer, matched case-insensitively at the
#                 start of a word, so "embed" matches "embedding" but "lose" does not match
#                 "close". An entry can be a list of alternatives where any one counts, for
#                 advice the documents phrase more than one way
```

Replace the `trim-vs-summarize` entry's `must_mention` line with:

```yaml
  # The doc says you "may lose information". Any form of the word counts.
  must_mention: [["lose", "lost", "loss"]]
```

Replace the `long-term-memory-store` entry's `must_mention` line with:

```yaml
  # "store" alone was in the question itself, so it passed on any answer that repeated it.
  # These name an actual store or how one is organised, and each appears in both expected files
  # except PostgresStore and BaseStore, which only add-memory.md uses.
  must_mention: [["InMemoryStore", "PostgresStore", "BaseStore", "namespace"]]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd $W && $PY -m unittest discover -s tests -v`
Expected: all tests pass (28 existing + 6 new = 34).

- [ ] **Step 6: Run the eval once to check nothing regressed (paid, under 1 cent)**

Run: `cd $MAIN && ( set -a; . ./.env; set +a; $PY $W/evals/run.py )`
Expected: `23/23 passed`. If a question newly fails, read its printed answer. If the answer is correct and the term was too strict, widen it with an any-of list and a comment saying why, then re-run only that question (`$PY $W/evals/run.py <id>`). If the answer is wrong, do not loosen the check: report it.

- [ ] **Step 7: Commit**

```bash
cd $W && git add evals/run.py evals/questions.yaml tests/test_eval_runner.py
git commit -m "Grade eval answers more fairly by matching terms at word starts and only counting a refusal when the answer is also missing"
```

---

### Task 2: Add near-miss refusal questions, and tighten the prompt only if they fail

The five refusal questions are off topic, which any model declines. A near miss asks about something the documents discuss at length but never answer. This is the "decline when the context does not actually answer" roadmap item, and the failure a small local model is expected to show.

**Files:**
- Modify: `evals/questions.yaml` (new section after the refusals)
- Modify: `tests/test_eval_runner.py` (count derived from the file, new guard test)
- Modify, only if Step 5 says so: `app.py` (`ANSWER_PROMPT` system text)
- Modify: `README.md` ("Evaluating it" counts), `ROADMAP.md` ("Where it stands" and the Phase 3 eval bullets)

**Interfaces:**
- Consumes: `grade()` and `is_refusal()` from Task 1, unchanged.
- Produces: 26 questions in `evals/questions.yaml`, every refusal id starting `refuse-`. Task 6 relies on both.

- [ ] **Step 1: Add the questions**

Append to `evals/questions.yaml`:

```yaml

# --- Must refuse: near misses ---
#
# The refusals above are off topic, and any model declines those easily. These ask about
# something the documents discuss at length but never answer, which is where a model fills the
# gap from its own training or from a code sample written for something else. That is what
# happened to chat-model-return-type before the chunk size was fixed: the wrong chunks came back
# and the model answered from an unrelated snippet instead of declining.

- id: refuse-near-miss-encrypt-checkpoints
  question: How do I encrypt the checkpoints that PostgresSaver stores?
  should_refuse: true
  # PostgresSaver setup appears throughout persistence.md and add-memory.md. Nothing in data/
  # mentions encryption.

- id: refuse-near-miss-redis-version
  question: Which Redis server version does RedisSaver require?
  should_refuse: true
  # add-memory.md shows RedisSaver setup and a connection string, never a server version.

- id: refuse-near-miss-embeddings-return-type
  question: What changed about the return type of embed_query in v1?
  should_refuse: true
  # Built to mirror chat-model-return-type. The migration guide covers the chat model return
  # type and lists embeddings modules, but never mentions embed_query.
```

- [ ] **Step 2: Run the tests to see the hard-coded count fail**

Run: `cd $W && $PY -m unittest tests.test_eval_runner -v`
Expected: FAIL in `test_refusal_questions_with_retrieved_docs_are_still_graded` with `8 != 5`.

- [ ] **Step 3: Derive the count from the file, and guard the id convention**

In `tests/test_eval_runner.py`, add `import yaml` after `from unittest import mock`. Replace the last line of `test_refusal_questions_with_retrieved_docs_are_still_graded` with:

```python
        questions = yaml.safe_load(eval_run.QUESTIONS_FILE.read_text())
        refusals = [q for q in questions if "refuse" in q["id"]]
        self.assertEqual(output.count("PASS"), len(refusals))
```

Add a new class above `MainTest`:

```python
class QuestionSetTest(unittest.TestCase):
    def test_every_refusal_question_can_be_selected_with_the_refuse_filter(self):
        # `python evals/run.py refuse` is how you check refusals alone after a prompt change.
        questions = yaml.safe_load(eval_run.QUESTIONS_FILE.read_text())
        for question in questions:
            if question.get("should_refuse"):
                self.assertTrue(question["id"].startswith("refuse-"), question["id"])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd $W && $PY -m unittest discover -s tests -v`
Expected: 35 tests, all pass.

- [ ] **Step 5: Measure the near misses with the current prompt (paid, under 1 cent)**

Run: `cd $MAIN && ( set -a; . ./.env; set +a; $PY $W/evals/run.py near-miss )`

- If `3/3 passed`: do not change the prompt. Go to Step 7. Record in the ROADMAP (Step 7) that `gpt-4o-mini` declines all three with the current prompt, and that the local run in Task 6 is the real test.
- If any fail: go to Step 6.

- [ ] **Step 6 (only if Step 5 had a failure): Tighten the prompt and measure again**

In `app.py`, inside `ANSWER_PROMPT`'s system text, replace the line `If the answer is not in the context, say you do not know.` with:

```text
If the answer is not in the context, say you do not know. The context often contains passages
on the same topic that still do not answer the question. Answer only if the context states the
answer itself. Do not fill a gap from your own knowledge, and do not adapt a code example that
was written for something else.
```

Re-run `near-miss`, then the full set: `cd $MAIN && ( set -a; . ./.env; set +a; $PY $W/evals/run.py )` (paid, under 1 cent). Keep the change only if the near misses improve and all 18 answerable questions still pass. If answerable questions start failing, revert the prompt edit and report both results instead of trying further wordings.

- [ ] **Step 7: Run the full set, then update the counts in the docs (paid, under 1 cent; skip the run if Step 6 already ran it)**

Run: `cd $MAIN && ( set -a; . ./.env; set +a; $PY $W/evals/run.py )`. Use its score below.

`README.md`, "Evaluating it": change the first sentence to say 26 questions: 18 the documents answer and 8 they don't, which the app must decline, three of them near misses on topics the documents discuss without answering.

`ROADMAP.md`: in "Where it stands", change "An eval set of 23 questions" and the score sentence to the new count and the score from this run, dated 2026-10-18 or the actual run date. In Phase 3, update the "Write the eval set first" bullet's counts to 26, 18 and 8. Update the "Make the prompt decline..." bullet with what Step 5 or 6 found, and tick it only if the near misses now pass. Remove the first two "Known issues" entries (refusal check, weak terms), since Task 1 fixed them, and add a sentence to the Phase 3 eval bullet saying the grading was tightened and why.

- [ ] **Step 8: Commit**

```bash
cd $W && git add evals/questions.yaml tests/test_eval_runner.py README.md ROADMAP.md
# add app.py too if Step 6 kept the prompt change
git commit -m "Add near-miss refusal questions that ask about topics the documents discuss but never answer"
```

---

### Task 3: Add `models.py`, the one place that chooses models

**Files:**
- Modify: `requirements.txt`
- Create: `models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces (Task 4 relies on these exact names):
  - `backend(setting: str) -> str` returns `"openai"` or `"ollama"`, raises `ValueError` otherwise
  - `embedding_model_name() -> str`, `chat_model_name() -> str`
  - `collection_name() -> str`, equal to the embedding model name
  - `make_embeddings()` returns `OpenAIEmbeddings` or `NomicEmbeddings`
  - `make_chat_model()` returns `ChatOpenAI` or `ChatOllama`
  - `setup_problem(include_chat: bool = True) -> str | None`
  - `describe() -> str`
  - constants `EMBEDDING_MODELS`, `CHAT_MODELS`, `OLLAMA_CONTEXT_TOKENS = 8192`, class `NomicEmbeddings`

- [ ] **Step 1: Add the dependency and install it**

In `requirements.txt`, add this line after `langchain-core==1.4.8`:

```text
langchain-ollama==1.1.0
```

Check the install would not move `langchain-core`, then install. This downloads from PyPI into the project's virtualenv only:

```bash
$PY -m pip install --dry-run langchain-ollama==1.1.0   # must not list langchain-core
$PY -m pip install langchain-ollama==1.1.0
$PY -m pip check
$PY -m pip show langchain-core | grep Version          # must still be 1.4.8
```

If the dry run wants to change `langchain-core`, stop and report.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_models.py`:

```python
import os
import unittest
from unittest import mock

from langchain_openai import ChatOpenAI

import models


def settings(**values):
    """Run with exactly these settings, whatever the real environment or .env says."""
    managed = ("EMBEDDING_BACKEND", "CHAT_BACKEND", "OPENAI_API_KEY")
    environ = {key: value for key, value in os.environ.items() if key not in managed}
    environ.update(values)
    return mock.patch.dict(os.environ, environ, clear=True)


class BackendTest(unittest.TestCase):
    def test_defaults_to_openai(self):
        with settings():
            self.assertEqual(models.backend("CHAT_BACKEND"), "openai")

    def test_is_case_insensitive_and_ignores_spaces(self):
        with settings(CHAT_BACKEND=" Ollama "):
            self.assertEqual(models.backend("CHAT_BACKEND"), "ollama")

    def test_rejects_an_unknown_backend(self):
        with settings(CHAT_BACKEND="olama"):
            with self.assertRaisesRegex(ValueError, "openai or ollama"):
                models.backend("CHAT_BACKEND")


class CollectionNameTest(unittest.TestCase):
    def test_each_embedding_backend_gets_its_own_collection(self):
        with settings(EMBEDDING_BACKEND="openai"):
            openai_collection = models.collection_name()
        with settings(EMBEDDING_BACKEND="ollama"):
            ollama_collection = models.collection_name()
        self.assertEqual(openai_collection, "text-embedding-3-small")
        self.assertEqual(ollama_collection, "nomic-embed-text")


class MakeModelsTest(unittest.TestCase):
    def test_openai_chat_model_is_deterministic(self):
        with settings(CHAT_BACKEND="openai", OPENAI_API_KEY="test"):
            model = models.make_chat_model()
        self.assertIsInstance(model, ChatOpenAI)
        self.assertEqual(model.model_name, "gpt-4o-mini")
        self.assertEqual(model.temperature, 0)

    def test_ollama_chat_model_sets_the_context_window(self):
        with settings(CHAT_BACKEND="ollama"), mock.patch.object(models, "ChatOllama") as chat:
            models.make_chat_model()
        chat.assert_called_once_with(
            model="llama3.2:3b", temperature=0, num_ctx=8192, validate_model_on_init=True
        )

    def test_stored_text_gets_the_document_prefix(self):
        embeddings = models.NomicEmbeddings(model="nomic-embed-text")
        with mock.patch.object(
            models.OllamaEmbeddings, "embed_documents", return_value=[[0.0]]
        ) as parent:
            embeddings.embed_documents(["Use a checkpointer."])
        parent.assert_called_once_with(["search_document: Use a checkpointer."])

    def test_questions_get_the_query_prefix(self):
        embeddings = models.NomicEmbeddings(model="nomic-embed-text")
        with mock.patch.object(
            models.OllamaEmbeddings, "embed_query", return_value=[0.0]
        ) as parent:
            embeddings.embed_query("What is a checkpointer?")
        parent.assert_called_once_with("search_query: What is a checkpointer?")


class SetupProblemTest(unittest.TestCase):
    def test_missing_key_is_reported_when_openai_is_used(self):
        with settings():
            self.assertIn("OPENAI_API_KEY", models.setup_problem())

    def test_no_key_needed_when_everything_runs_locally(self):
        with settings(EMBEDDING_BACKEND="ollama", CHAT_BACKEND="ollama"), \
             mock.patch.object(models, "NomicEmbeddings"), \
             mock.patch.object(models, "ChatOllama"):
            self.assertIsNone(models.setup_problem())

    def test_ollama_not_running_is_reported(self):
        failure = ValueError("Failed to connect to Ollama.")
        with settings(EMBEDDING_BACKEND="ollama", OPENAI_API_KEY="test"), \
             mock.patch.object(models, "NomicEmbeddings", side_effect=failure):
            self.assertIn("Failed to connect to Ollama", models.setup_problem())

    def test_ingest_does_not_need_the_chat_model(self):
        failure = ValueError("Model `llama3.2:3b` not found in Ollama.")
        with settings(EMBEDDING_BACKEND="openai", CHAT_BACKEND="ollama", OPENAI_API_KEY="test"), \
             mock.patch.object(models, "ChatOllama", side_effect=failure):
            self.assertIsNone(models.setup_problem(include_chat=False))

    def test_unknown_backend_is_reported_not_raised(self):
        with settings(CHAT_BACKEND="gpt", OPENAI_API_KEY="test"):
            self.assertIn("openai or ollama", models.setup_problem())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cd $W && $PY -m unittest tests.test_models -v`
Expected: ERROR, `ModuleNotFoundError: No module named 'models'`.

- [ ] **Step 4: Write `models.py`**

```python
"""Which models embed text and write answers, chosen in one place.

ingest.py embeds the documents and app.py embeds each question, and the two must use the same
embedding model or retrieval compares vectors that mean nothing to each other. Before this file
each script named the model itself, with a comment asking you to keep the two in sync. Now both
ask this module.

Two settings, read from the environment or .env, each "openai" or "ollama":

    EMBEDDING_BACKEND  turns documents and questions into vectors
    CHAT_BACKEND       rewrites follow-up questions and writes the answers

They are separate because they are separate jobs, and mixing them shows which half of RAG
drives answer quality.
"""

import os

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


BACKENDS = ("openai", "ollama")

# Named explicitly, never left to a library default. The old OpenAI default,
# text-embedding-ada-002, has the same 1536 dimensions as text-embedding-3-small, so a mismatch
# between indexing and searching raises nothing and quietly retrieves wrong chunks.
#
# Changing the Ollama model means changing NomicEmbeddings too: its prefixes belong to
# nomic-embed-text.
EMBEDDING_MODELS = {
    "openai": "text-embedding-3-small",
    "ollama": "nomic-embed-text",
}

CHAT_MODELS = {
    "openai": "gpt-4o-mini",
    # A 2.0 GB download that leaves room for the embedding model and everything else on an
    # 8 GB Mac. Larger models run, but slowly, once they push the machine into swapping.
    "ollama": "llama3.2:3b",
}

# Ollama gives a model a 4096-token window unless told otherwise, on machines with less than
# 24 GiB of GPU memory. Six retrieved chunks, the instructions, and six messages of history can
# get close to that, and what Ollama does with a prompt that overflows is not something this app
# should depend on. Twice the default keeps every prompt well inside.
OLLAMA_CONTEXT_TOKENS = 8192


class NomicEmbeddings(OllamaEmbeddings):
    """nomic-embed-text, with the task prefixes its model card requires.

    The model was trained to see "search_document: " in front of text being stored and
    "search_query: " in front of a question, and its card says the prefix must be included.
    Neither Ollama nor OllamaEmbeddings adds it, and leaving it out raises no error, so nothing
    would tell you it was missing.
    """

    def embed_documents(self, texts):
        return super().embed_documents([f"search_document: {text}" for text in texts])

    def embed_query(self, text):
        return super().embed_query(f"search_query: {text}")


def backend(setting):
    """Read EMBEDDING_BACKEND or CHAT_BACKEND, defaulting to openai.

    Read when called rather than at import, so .env has been loaded by then and tests can
    change it.
    """
    value = os.getenv(setting, "openai").strip().lower()
    if value not in BACKENDS:
        raise ValueError(f"{setting} must be openai or ollama, not {value!r}.")
    return value


def embedding_model_name():
    return EMBEDDING_MODELS[backend("EMBEDDING_BACKEND")]


def chat_model_name():
    return CHAT_MODELS[backend("CHAT_BACKEND")]


def collection_name():
    """The Chroma collection that holds vectors from the current embedding model.

    One collection per model, so both indexes sit in vectorstore/ side by side. Switching
    EMBEDDING_BACKEND without re-indexing then finds an empty collection, which the app reports
    as "run ingest.py", instead of comparing a question against another model's vectors.
    """
    return embedding_model_name()


def make_embeddings():
    if backend("EMBEDDING_BACKEND") == "ollama":
        # validate_model_on_init checks that Ollama is running and the model is pulled, and
        # raises a ValueError saying which, instead of a connection error on the first search.
        return NomicEmbeddings(model=EMBEDDING_MODELS["ollama"], validate_model_on_init=True)
    return OpenAIEmbeddings(model=EMBEDDING_MODELS["openai"])


def make_chat_model():
    # temperature=0 on both, so the same question gets the same answer and eval runs repeat.
    if backend("CHAT_BACKEND") == "ollama":
        return ChatOllama(
            model=CHAT_MODELS["ollama"],
            temperature=0,
            num_ctx=OLLAMA_CONTEXT_TOKENS,
            validate_model_on_init=True,
        )
    return ChatOpenAI(model=CHAT_MODELS["openai"], temperature=0)


def setup_problem(include_chat=True):
    """Say what is missing before any model is called, or return None if nothing is.

    Without this, a missing key or a stopped Ollama shows up as a traceback on the first
    question. ingest.py passes include_chat=False because it never calls the chat model.
    """
    try:
        settings = ["EMBEDDING_BACKEND"] + (["CHAT_BACKEND"] if include_chat else [])
        in_use = {backend(setting) for setting in settings}

        if "openai" in in_use and not os.getenv("OPENAI_API_KEY"):
            return (
                "No OPENAI_API_KEY found. Copy .env.example to .env and add your key, or set "
                "EMBEDDING_BACKEND and CHAT_BACKEND to ollama to run without one."
            )

        # Creating an Ollama model checks the server and the model, see make_embeddings().
        if backend("EMBEDDING_BACKEND") == "ollama":
            make_embeddings()
        if include_chat and backend("CHAT_BACKEND") == "ollama":
            make_chat_model()
    except ValueError as error:
        return str(error)
    return None


def describe():
    """One line naming the models in use, for the app caption and the top of an eval run."""
    return f"Answers from {chat_model_name()}, search with {embedding_model_name()}"
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd $W && $PY -m unittest discover -s tests -v`
Expected: all pass (35 from Task 2 + 13 new = 48). If patching a method on the Pydantic class `OllamaEmbeddings` fails, report it rather than rewriting the tests to reach into private attributes.

- [ ] **Step 6: Commit**

```bash
cd $W && git add requirements.txt models.py tests/test_models.py
git commit -m "Add models.py so the embedding and chat backends are chosen in one place, with Ollama as an option"
```

---

### Task 4: Wire `models.py` into ingest, the app, and the eval runner

**Files:**
- Modify: `ingest.py` (imports, remove `EMBEDDING_MODEL`, `main()`)
- Modify: `app.py` (imports, remove `CHAT_MODEL` and `EMBEDDING_MODEL`, `load_retriever`, `build_search_query`, `answer_question`, `main`)
- Modify: `evals/run.py` (setup check, model line, timing)
- Modify: `.env.example`
- Test: `tests/test_ingest.py`, `tests/test_app.py`, `tests/test_eval_runner.py`

**Interfaces:**
- Consumes from Task 3: `models.setup_problem(include_chat=...)`, `models.collection_name()`, `models.make_embeddings()`, `models.make_chat_model()`, `models.embedding_model_name()`, `models.describe()`.
- Produces: `answer_question(question, history=None) -> (answer, docs, search_query)`, unchanged. `evals/run.py` prints `models.describe()` first and `<N> s in total, <X> s per question` last. Task 6 reads both lines.

- [ ] **Step 1: Write the failing tests**

`tests/test_ingest.py`: add `import os` to the imports. Replace `run_main_with` and add two tests:

```python
    def run_main_with(self, files, problem=None):
        """Run ingest.main() against a temporary data folder, with Chroma and the models mocked out."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            for name, text in files.items():
                (data_dir / name).write_text(text)
            with mock.patch.object(ingest, "DATA_DIR", data_dir), \
                 mock.patch.object(ingest, "VECTORSTORE_DIR", Path(tmp) / "vectorstore"), \
                 mock.patch.object(ingest, "Chroma") as chroma, \
                 mock.patch.object(ingest.models, "make_embeddings"), \
                 mock.patch.object(ingest.models, "setup_problem", return_value=problem):
                # Kept on self so a test can inspect it even when main() raises.
                self.chroma = chroma
                # main() prints progress lines; keep them out of the test output.
                with contextlib.redirect_stdout(io.StringIO()):
                    ingest.main()
            return chroma

    def test_a_setup_problem_stops_before_touching_the_index(self):
        with self.assertRaisesRegex(SystemExit, "No OPENAI_API_KEY"):
            self.run_main_with({"real.md": "Text."}, problem="No OPENAI_API_KEY found.")
        self.chroma.assert_not_called()

    def test_each_embedding_model_writes_its_own_collection(self):
        with mock.patch.dict(os.environ, {"EMBEDDING_BACKEND": "ollama"}):
            chroma = self.run_main_with({"real.md": "# Memory\nUse a checkpointer."})
        self.assertEqual(chroma.call_args.kwargs["collection_name"], "nomic-embed-text")
```

`tests/test_app.py`: add `import os` and `from unittest import mock` to the imports, and `from langchain_core.messages import AIMessage`. Add:

```python
class WiringTest(unittest.TestCase):
    def test_retriever_reads_the_collection_for_the_current_embedding_model(self):
        with mock.patch.dict(os.environ, {"EMBEDDING_BACKEND": "ollama"}), \
             mock.patch.object(app, "Chroma") as chroma, \
             mock.patch.object(app.models, "make_embeddings"):
            app.load_retriever()
        self.assertEqual(chroma.call_args.kwargs["collection_name"], "nomic-embed-text")

    def test_answers_come_from_the_configured_chat_model(self):
        retriever = mock.Mock()
        retriever.invoke.return_value = [Document("Use a checkpointer.", metadata={"source": "data/a.md"})]
        chat = mock.Mock()
        chat.invoke.return_value = AIMessage("Use a checkpointer.")
        with mock.patch.object(app, "load_retriever", return_value=retriever), \
             mock.patch.object(app.models, "make_chat_model", return_value=chat):
            answer, docs, search_query = app.answer_question("How do I add memory?")
        self.assertEqual(answer, "Use a checkpointer.")
        self.assertEqual(search_query, "How do I add memory?")
        chat.invoke.assert_called_once()
```

`tests/test_eval_runner.py`: in `MainTest.run_main`, add `mock.patch.object(eval_run.models, "setup_problem", return_value=None), \` to the `with` block, then add:

```python
    def test_a_setup_problem_stops_before_any_question(self):
        output = io.StringIO()
        with mock.patch.object(eval_run.models, "setup_problem", return_value="No OPENAI_API_KEY found."), \
             mock.patch.object(eval_run, "answer_question") as answer_question, \
             mock.patch.object(eval_run.sys, "argv", ["run.py"]), \
             redirect_stdout(output):
            exit_code = eval_run.main()
        self.assertEqual(exit_code, 1)
        self.assertIn("No OPENAI_API_KEY found.", output.getvalue())
        answer_question.assert_not_called()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd $W && $PY -m unittest discover -s tests -v`
Expected: errors and failures mentioning `models` (`ingest`, `app` and `eval_run` have no attribute `models`).

- [ ] **Step 3: Change `ingest.py`**

- Replace `from langchain_openai import OpenAIEmbeddings` with nothing, and add `import models` as its own group after the third-party imports (a blank line, then `import models`).
- Delete the `EMBEDDING_MODEL` constant and its comment block. `models.py` now carries that note.
- In `main()`, right after `load_dotenv()`, add:

```python
    # Checked before anything is read or deleted. ingest.py only embeds, so the chat backend
    # does not need to be set up for it.
    problem = models.setup_problem(include_chat=False)
    if problem:
        raise SystemExit(problem)
```

- Replace the `Chroma(...)` call with:

```python
    vectorstore = Chroma(
        collection_name=models.collection_name(),
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=models.make_embeddings(),
    )
```

- Replace the two `print` lines at the end with:

```python
    print(f"Indexed {len(chunks)} chunks from {len(documents)} document pages/files.")
    print(f"Embedded with {models.embedding_model_name()}, saved to {VECTORSTORE_DIR}/")
```

- [ ] **Step 4: Change `app.py`**

- Delete `from langchain_openai import ChatOpenAI, OpenAIEmbeddings`. Add `import models` as its own group after the third-party imports.
- Delete `CHAT_MODEL = "gpt-4o-mini"` and the `EMBEDDING_MODEL` constant with its comment.
- In `load_retriever()`, replace the body after the comment with:

```python
    vectorstore = Chroma(
        collection_name=models.collection_name(),
        persist_directory=VECTORSTORE_DIR,
        embedding_function=models.make_embeddings(),
    )
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
```

- In `build_search_query()` and `answer_question()`, replace each `ChatOpenAI(model=CHAT_MODEL, temperature=0)` with `models.make_chat_model()`.
- In `main()`, replace the `st.caption(...)` line and the whole `OPENAI_API_KEY` check (comment included) with:

```python
    # Checked up front. Otherwise a missing key or a stopped Ollama surfaces as a raw traceback
    # on the first question, because no model is created until then.
    problem = models.setup_problem()
    if problem:
        st.error(problem)
        st.stop()

    st.caption(
        f"Ask questions about the documents indexed from the data folder. {models.describe()}."
    )
```

- [ ] **Step 5: Change `evals/run.py`**

- Add `import time` to the imports (after `import sys`). After the `from app import ...` line, add `import models  # noqa: E402`.
- Update the module docstring's cost paragraph to say each question costs one embedding call and one chat call on OpenAI, and nothing on Ollama.
- In `main()`, after the question filter and before `failures = []`, add:

```python
    problem = models.setup_problem()
    if problem:
        print(problem)
        return 1

    # Printed first so a saved run says which models produced it.
    print(models.describe())
    print()

    started = time.perf_counter()
```

- After the loop, before `total = len(questions)`, add `elapsed = time.perf_counter() - started`. After the `failed:` print, add:

```python
    # Speed is half of what the local comparison measures. Wall-clock time, so it includes
    # the network for OpenAI and the first model load for Ollama.
    print(f"{elapsed:.0f} s in total, {elapsed / total:.1f} s per question")
```

- [ ] **Step 6: Document the settings in `.env.example`**

Replace its contents with:

```text
OPENAI_API_KEY=your_api_key_here

# Which models embed text and write answers: openai or ollama. Mixing them is allowed.
# Each embedding backend has its own index, so run `python ingest.py` after changing
# EMBEDDING_BACKEND. Ollama needs to be installed and running; see the README.
EMBEDDING_BACKEND=openai
CHAT_BACKEND=openai
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `cd $W && $PY -m unittest discover -s tests -v`
Expected: all pass (48 + 5 new = 53). Also confirm no stale names remain: `cd $W && grep -n -E "CHAT_MODEL|EMBEDDING_MODEL\b|OpenAIEmbeddings|ChatOpenAI" app.py ingest.py evals/run.py` prints nothing.

- [ ] **Step 8: Re-index and re-run the OpenAI eval (paid, under 1 cent)**

The OpenAI index moves from Chroma's default collection name to `text-embedding-3-small`, so it must be rebuilt once. The old collection stays in `vectorstore/` unused and harmless.

```bash
cd $MAIN && ( set -a; . ./.env; set +a; $PY $W/ingest.py )
cd $MAIN && ( set -a; . ./.env; set +a; $PY $W/evals/run.py )
```

Expected: ingest reports 320 chunks embedded with `text-embedding-3-small`. The eval prints `Answers from gpt-4o-mini, search with text-embedding-3-small` first, the same score as Task 2's final run, and a timing line last.

- [ ] **Step 9: Commit**

```bash
cd $W && git add ingest.py app.py evals/run.py .env.example tests/
git commit -m "Build every model through models.py, keep one index per embedding model, and time eval runs"
```

---

### Task 5: Install Ollama and pull the models (the owner does this, not a subagent)

This installs software on the Mac, so the owner runs or approves each command.

- [ ] **Step 1: Install and start the server**

```bash
brew install ollama          # Homebrew formula, 0.40.6 as of 2026-10-18
ollama serve                 # leave running in its own terminal tab; Ctrl+C stops it
```

Running `ollama serve` in a tab, rather than as a background service, keeps it visible and easy to stop, and shows its log while the eval runs.

- [ ] **Step 2: Pull the two models (about 2.3 GB in total)**

```bash
ollama pull llama3.2:3b       # 2.0 GB
ollama pull nomic-embed-text  # 274 MB
ollama list
```

Expected: both models listed.

- [ ] **Step 3: Smoke test**

```bash
ollama run llama3.2:3b "Reply with the single word: ready"
```

Expected: a short reply within a minute. The first run is slower because the model loads into memory.

---

### Task 6: Run the comparison (the controller runs this, needs Task 5)

**Files:**
- Create: `evals/results/2026-10-DD-embed-<E>-chat-<C>.txt`, four files (use the actual run date)

- [ ] **Step 1: Build the Ollama index**

```bash
cd $MAIN && ( set -a; . ./.env; set +a; time EMBEDDING_BACKEND=ollama $PY $W/ingest.py )
```

Expected: 320 chunks embedded with `nomic-embed-text`, the same count as the OpenAI index, since chunking does not depend on the model. Note the time for the README.

- [ ] **Step 2: Check the prompt fits in Ollama's window**

Save this as `context_check.py` in the session scratchpad (`$SCRATCH` below):

```python
import sys
sys.path.insert(0, "/Users/erinrose/.superset/worktrees/chatbot/quaint-biology-92a8aeba")

import app
import models

question = "What is the difference between a checkpointer and a store?"
# Six long messages, the most history the app ever sends.
history = [
    {"role": "user" if i % 2 == 0 else "assistant", "content": "A long earlier turn. " * 40}
    for i in range(app.MAX_HISTORY_MESSAGES)
]
docs = app.load_retriever().invoke(question)
messages = app.ANSWER_PROMPT.format_messages(
    history=app.to_messages(history), context=app.format_docs(docs), question=question
)
response = models.make_chat_model().invoke(messages)
print("input tokens:", response.usage_metadata["input_tokens"], "of", models.OLLAMA_CONTEXT_TOKENS)
```

Run it from the main checkout, so `vectorstore/` resolves there:

```bash
cd $MAIN && ( set -a; . ./.env; set +a; EMBEDDING_BACKEND=ollama CHAT_BACKEND=ollama $PY $SCRATCH/context_check.py )
```

Then run `ollama ps` and check the `CONTEXT` column shows 8192. Expected: input tokens well under 8192. If they are over about 6000, report before running the eval.

- [ ] **Step 3: Run the four combinations**

Run each one separately, because `run.py` exits non-zero when any question fails. The two runs that use OpenAI are paid, under 1 cent each. The local chat runs take several minutes each.

```bash
mkdir -p $W/evals/results
cd $MAIN && set -a && . ./.env && set +a
for E in openai ollama; do
  for C in openai ollama; do
    EMBEDDING_BACKEND=$E CHAT_BACKEND=$C $PY $W/evals/run.py \
      | tee "$W/evals/results/$(date +%F)-embed-$E-chat-$C.txt"
  done
done
```

Run it in the background if a single combination exceeds the 10-minute tool timeout.

- [ ] **Step 4: Read every failure before summarising**

For each file, list the failed ids and sort each failure into: retrieval miss (`never retrieved`), wrong or incomplete answer (`missing:`), refused when it should answer, or answered when it should refuse. Check at least two failing answers by hand against the documents, to confirm the grader is right and not the model. Write the findings down for Task 7. They are the substance of the comparison.

- [ ] **Step 5: Commit the raw results**

```bash
cd $W && git add evals/results/
git commit -m "Record eval runs for all four combinations of OpenAI and Ollama embeddings and chat models"
```

---

### Task 7: Write up the comparison

**Files:**
- Modify: `README.md`
- Modify: `ROADMAP.md`

Every number in this task comes from the Task 6 result files and notes. Do not estimate a number the runs measured.

- [ ] **Step 1: README, setup for running locally**

After the existing Setup commands, add a subsection "Running it locally with Ollama". Give the Task 5 install and pull commands. Say to set `EMBEDDING_BACKEND` and `CHAT_BACKEND` in `.env`, to run `python ingest.py` after changing the embedding backend, and that no OpenAI key is needed when both are `ollama`. Mention that an 8 GB Mac is enough for these two models and that answers are slower.

- [ ] **Step 2: README, the comparison table**

Add a section "OpenAI against a local model" after "Evaluating it", with one row per result file:

| Embeddings | Answers | Score (of 26) | Answerable (of 18) | Refusals (of 8) | Seconds per question | Cost per run |
|---|---|---|---|---|---|---|

- Score: from the `N/26 passed` line.
- Answerable and Refusals: count the failed ids in each group.
- Seconds per question: from the timing line.
- Cost per run: "under 1 cent" when answers come from OpenAI. "Effectively nothing" for OpenAI embeddings with local answers (26 query embeddings). "$0" for fully local. Footnote that these are estimates from token counts, as in the Cost section.

Below the table, two to four sentences on which half drove the differences, using the Task 6 Step 4 findings. Link the four result files.

- [ ] **Step 3: README, the other sections that change**

- "How it works": steps 3 and 7 name the models. Say the embedding and chat models are each OpenAI or Ollama, chosen in `models.py`.
- "Cost": add one sentence that the local backend costs nothing per question but is slower, with the measured number.
- "Project layout": add `models.py` (which models embed and answer) and `evals/results/` (saved eval runs).
- "What I learned": add entries only for what the runs showed. Candidates, kept only if the data supports them: which half of RAG drove quality; the nomic prefixes being required and silent when missing; Ollama's 4096-token default window.
- "Next steps": the Ollama comparison is done. Point to the roadmap's remaining items.

- [ ] **Step 4: ROADMAP**

- "Last updated" and "Where it stands": the comparison is done, with a one-line result.
- Phase 3: tick each remaining item with a short note of what was done and measured. The "Re-ingest when switching embedding models" item became one collection per embedding model. Say so, and why.
- Change the "(next)" marker on Phase 3 to "(done)".
- "Verification notes": add that the nomic prefixes were added because the model card requires them but were not A/B tested without them; that Ollama's behaviour on an overflowing prompt was avoided, not tested; and that the speed figures come from one Apple M2 with 8 GB.
- "Known issues": add anything Task 6 found that is left unfixed.

- [ ] **Step 5: Check and commit**

Run: `cd $W && $PY -m unittest discover -s tests -v` (all pass). Then check for em dashes, which must not appear: `grep -n "$(printf '\xe2\x80\x94')" README.md ROADMAP.md` must print nothing.

```bash
cd $W && git add README.md ROADMAP.md
git commit -m "Write up the OpenAI and Ollama comparison in the README and close out Phase 3 in the roadmap"
```

---

## After the last task

Run the final whole-branch review, then use superpowers:finishing-a-development-branch. Merging to `main` is the owner's decision. Nothing is pushed.
