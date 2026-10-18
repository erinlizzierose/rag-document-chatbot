# Roadmap

This project exists to understand how retrieval augmented generation actually works, by
building one small enough to read in an afternoon. This file tracks what has been built,
what is next, and what was learned along the way. For setup and usage, see the
[README](README.md).

Last updated 2026-10-18.

## Where it stands

The app works end to end. It answers questions from the documents in `data/`, shows the
passages it used, declines when the documents don't cover something, and handles follow-up
questions by rewriting them to stand alone before searching. An eval set of 26 questions
measures it, and the current settings scored 25 of 26 when last run (2026-10-18). The one
miss is `long-term-memory-store`, explained under Phase 3. Offline unit tests cover the
parts that don't call a model.

Next is running the same pipeline against a local model and comparing the two.

## Phase 0: Make it real (done)

- [x] **Replace the placeholder document with real ones.** The original sample file was 927
      characters, a single chunk, so the retriever handed over the whole document every time
      and retrieval did nothing. Five LangChain documentation pages replaced it.
- [x] **Check that retrieval actually chooses.** A question the documents answer pulled
      chunks from only two of the files, which is retrieval doing its job.
- [x] **Check that it declines.** Asked for the capital of France, it said it did not know.
      *Retrieval still returned chunks, because similarity search has no concept of "no
      match." The grounding instruction in the prompt is what turns that into an honest
      refusal.*

## Phase 1: Correctness (done)

- [x] **Make re-indexing safe.** `Chroma.from_documents` generated a fresh ID per chunk
      against a get-or-create collection, so every run appended a second copy and paid for
      the embeddings again. `ingest.py` now clears the collection before adding.
      *Verified: a second ingest left 240 vectors, not 480.*
- [x] **Name the embedding model explicitly in both files.** The old default,
      `text-embedding-ada-002`, has the same 1536 dimensions as `text-embedding-3-small`, so
      a mismatch raises nothing and quietly returns bad matches.
- [x] **Guard the empty index.** An empty store returns no matches rather than an error. The
      app checks for that and tells you to run `ingest.py`, before paying for the answer
      call.
- [x] **Friendlier missing-key error.** The app checks for `OPENAI_API_KEY` at startup
      instead of failing with a traceback on the first question.
- [x] **Move retrieved text out of the system message.** Retrieved chunks sit in the user
      turn inside `<context>` tags, and the system prompt says tagged text is untrusted
      content, never instructions.
- [x] **Strip agent-directed boilerplate during ingest.** The documentation pages opened with
      a note addressed to an LLM. `strip_leading_blockquote()` removes only that leading
      block, so real blockquotes further down survive. *Measured: on "summarize everything
      you know about memory," 2 of 4 retrieved chunks used to be boilerplate. Now 0 of 4.*
- [x] **Rewrite the README** to lead with what the project does, state real costs, and record
      what was learned.
- [x] **Add a screenshot** at `docs/screenshot.png`.

## Phase 2: Polish (done)

- [x] **Pin `requirements.txt` by hand**, not with `pip freeze`, so it lists only what the
      code imports. Bare `langchain` is left out because nothing imports it.
- [x] **Add an MIT `LICENSE`.**
- [x] **Attribute the bundled documents.** They are LangChain documentation, MIT licensed via
      the [langchain-ai/docs](https://github.com/langchain-ai/docs) repository.
      [`data/SOURCES.md`](data/SOURCES.md) lists each source URL, the fetch date, and the
      full upstream license. It is deliberately not indexed: as two chunks of citation table,
      it competed with real content on vague questions.
- [x] **Remove the stale sample file**, whose own text said it existed only until real
      documents were added.

## Pre-release fixes (done)

From a review of the whole history before publishing.

- [x] **Re-indexing no longer breaks a running app.** The retriever was cached for the life
      of the server, and `ingest.py` replaces the collection, so after re-indexing every
      question failed with "Collection does not exist" until a restart, even though the app
      said to reload. The retriever is now opened per question.
- [x] **Sources stay on earlier answers.** Streamlit redraws the whole page from saved
      history on every question, and only the answer text was saved.
- [x] **An ingest that finds no text stops before deleting the existing index**, instead of
      wiping it and then crashing.
- [x] **Document text can no longer close the `<context>` block.** The bundled docs contain
      their own example prompts with literal `</context>` tags.
- [x] **The eval runner refuses to score an empty index**, instead of counting the "run
      ingest.py" message as five correct refusals.
- [x] **Offline unit tests** for everything that doesn't call a model:
      `python -m unittest discover -s tests -v`.

## Phase 3: Run it locally too, and compare (next)

The step that makes this more than a tutorial clone: the same pipeline on two backends,
measured.

- [x] **Write the eval set first.** [`evals/questions.yaml`](evals/questions.yaml) holds 26
      questions, 18 the documents answer and 8 the app must decline.
      [`evals/run.py`](evals/run.py) checks answer text and retrieved sources separately,
      because retrieval usually degrades before answers visibly do. *Its first real use was
      choosing how many chunks to retrieve: k=4 scored 21/23 and k=6, 8, and 10 each scored
      22/23, so the app moved from 4 to 6.* The grading was later tightened because two
      checks were too easy to pass: terms are now matched at the start of a word, so "close"
      no longer satisfies "lose", and a refusal only counts against an answerable question
      when its terms are also missing, so a correct answer with a caveat is not marked wrong.
      Under the stricter terms `long-term-memory-store` now fails (25/26 on 2026-10-18). It
      had been passing only because "store" appears in its own question. The answer names no
      store because `langchain-add-memory.md` is not retrieved for that question.
- [x] **Find out why one question still fails at k=6.** It is `chat-model-return-type`, and it
      is a real retrieval miss, not a wrong expectation: `langchain-v1-migration.md` says the
      return type changed from `BaseMessage` to `AIMessage`, but five of the six retrieved
      chunks came from `langchain-add-memory.md`. *The model then answered confidently from an
      unrelated code snippet instead of declining, so a near-miss in retrieval became a wrong
      answer rather than an honest "I don't know."* Re-run on 2026-10-13 after the pre-release
      fixes: still 22/23, all five refusals held.
- [x] **Fix that retrieval miss, measured with the eval.** The answer chunk ranked 15th
      because the short section shared a 1000-character chunk with unrelated text. Tested one
      change at a time on retrieval alone: stripping links and markup made it worse (rank 42,
      the boundaries shifted), dropping duplicate chunks barely helped (rank 12), and smaller
      chunks fixed it (rank 1). Chunk size is now 750. *The fix exposed an eval check that had
      been passing by luck: `checkpoints-unbounded` required "delete", but the documents say
      "prune". `must_mention` entries can now be a list of alternatives.* 23/23 on 2026-10-14.
- [x] **Make the prompt decline when the context does not actually answer.** With the wrong
      chunks, the model answered from an unrelated code snippet instead of saying it did not
      know. Retrieval no longer triggers that here, so three near-miss questions now test it:
      topics the documents discuss at length but never answer. `gpt-4o-mini` declines all
      three with the current prompt, so the prompt was left alone. The local model run is the
      real test.
- [ ] **Install Ollama** and pull a chat model and `nomic-embed-text`. Add `langchain-ollama`
      to `requirements.txt`, pinned to a version compatible with the pinned `langchain-core`.
- [ ] **Make the backend configurable, with the embedding model and chat model as separate
      settings.** Local embeddings with an API chat model is a reasonable combination, and it
      shows which half of RAG drives answer quality. The embedding setting must change in
      `ingest.py` and `app.py` together.
- [ ] **Re-ingest when switching embedding models.** Different models produce different
      vector sizes, so an index built with one is unusable with the other.
- [ ] **Run the eval set against each combination** and record answer quality, speed, and
      cost.
- [ ] **Put the comparison table in the README.**

Expectation going in: Ollama is free and works offline, but slower and weaker at obeying the
"answer only from context" instruction. The eval set's eight refusal questions are there to
measure exactly that.

## Phase 4: Stretch

- [x] **Conversation memory.** Follow-up questions are rewritten into standalone ones before
      retrieval, with one extra `gpt-4o-mini` call, because retrieval happens before the
      model sees anything and "the second one" is useless as a search string. History is
      capped at six messages. *Bug found along the way: sending the rewritten question to
      the retriever but the raw one to the answering model let the two resolve the same
      pronoun differently. Right documents fetched, wrong question answered, no error
      anywhere. Both now get the rewritten question.* History lives in the Streamlit
      session, so it resets on page refresh.
- [ ] **Document upload in the app.** Adding a document currently means copying a file and
      running a terminal command. This is the gap between "works for me" and "usable by
      others."
- [x] **Compare chunk sizes.** Done while fixing the Phase 3 retrieval miss, on retrieval
      alone: 1000 left 17/18 answerable questions with the right text in the top 6 chunks,
      750 and 500 got 18/18, and 300 dropped to 16/18. Settled on 750, the largest that works.

### Deliberately not done

Framework-native memory. `langchain.memory` was removed in LangChain 1.0, and
`RunnableWithMessageHistory` warns as deprecated since langchain-core 1.3.3. The official
route is `create_agent` plus a LangGraph checkpointer, which is a large detour. Rewriting
the question by hand teaches more for a project this size.

## Known issues

Small problems found in the 2026-10-12 review and left for later. Each is a reasonable first
exercise.

- **`strip_leading_blockquote()` strips any leading blockquote**, so your own document that
  opens with a quote or epigraph loses it.
- **Paths are relative to the current folder.** Run the scripts from anywhere but the repo
  root and they look for `data/` and `vectorstore/` in the wrong place. On Windows, source
  paths use backslashes, so every `expect_source` check in the evals fails.
- **A failed OpenAI call leaves a question without an answer.** The app shows a traceback,
  and the unanswered question stays in the history sent with the next one.
- **The Sources panel renders chunks as Markdown and cuts them at 800 characters**, so
  headings appear oversized and long passages are shortened.

## Model choice

Staying on `gpt-4o-mini`. As of 2026-10-12, OpenAI's deprecations page lists no shutdown
date for it, and it accepts `temperature=0`, which the app relies on for repeatable answers.

Worth knowing before switching: `langchain-openai` silently discards `temperature` for model
names starting with `gpt-5`, unless the name contains `chat` or reasoning effort is `none`.
Moving to a `gpt-5.x` model would quietly remove that determinism.

## Verification notes

What has and has not been checked, so an assumption is not mistaken for a test.

- `gpt-4o-mini` prices were re-checked on 2026-10-12 against OpenAI's model page. The
  embedding price comes from a third-party listing. Cost figures in the README are estimates
  from token counts, not measured bills.
- The PDF branch of `load_documents()` has only been exercised offline with a blank PDF, never
  with a real document through the app. The bundled corpus is all Markdown.
- The re-indexing fix was verified offline with fake embeddings, not by re-indexing under a
  live Streamlit server.
- The claim that two 1536-dimension embedding models fail silently together is inferred from
  Chroma checking only vector width. It has not been reproduced with two live models.
- Code review, 2026-08-05: no secrets in `data/`, and `.env` is gitignored. The documents are
  full of `{"configurable": {"thread_id": "1"}}` samples and `ChatPromptTemplate` uses
  f-string syntax, so a runtime `KeyError` looked likely. Tested rather than assumed:
  substitution is single-pass and braces in values stay literal. Not a bug.
