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

The same pipeline also runs on local Ollama models. All four combinations of OpenAI and Ollama
embeddings and answers were evaluated on 2026-10-18 and scored 23 to 25 of 26. Local answers
were roughly seven to nine times slower (one Apple M2 with 8 GB, wall-clock including network
and model loading). The fully local pair scored lowest: the small model left a key out of one
answer and declined another question the documents answer. The two `nomic-embed-text` runs were
redone the same day after a bug in its question prefix was found and fixed. The README has the
table. What is left is the stretch work in Phase 4 and the open issues below.

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
      a mismatch raises nothing and quietly returns bad matches. *Since moved to `models.py`
      in Phase 3, so only one file names it.*
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

## Phase 3: Run it locally too, and compare (done)

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
      store, and none of the retrieved chunks gives it one: OpenAI embeddings fetch nothing
      from `langchain-add-memory.md` for that question, and `nomic-embed-text` fetches two
      chunks of it that are about short-term memory, not stores.
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
      three with the current prompt, so the prompt was left alone. The local model was the
      real test: `llama3.2:3b` declined all 8 refusal questions, including the 3 near misses,
      with the same prompt. *The refusal check is a marker list, so "The context does not
      mention X, but X is Y" would pass it. The eval now prints every refusal answer, and the
      saved runs show all 32 across the four combinations were refusals. All 16 from
      `gpt-4o-mini` were "I do not know."; of the 16 from `llama3.2:3b`, 11 were "I don't
      know.", 3 "I do not know." and 2 "You do not know."*
- [x] **Install Ollama** and pull a chat model and `nomic-embed-text`. Ollama 0.40.6 from
      Homebrew, `llama3.2:3b` (2.0 GB download) and `nomic-embed-text` (274 MB).
      `langchain-ollama==1.1.0` is pinned in `requirements.txt`, compatible with the pinned
      `langchain-core`.
- [x] **Make the backend configurable, with the embedding model and chat model as separate
      settings.** `models.py` reads `EMBEDDING_BACKEND` and `CHAT_BACKEND`, each `openai` or
      `ollama`, and is the one place that names the models. `ingest.py`, `app.py` and
      `evals/run.py` all build their models through it. `nomic-embed-text` needs a task prefix
      on stored text and on questions, which `models.py` adds. *A review found that every
      question had been getting the document prefix as well as the query prefix: the parent
      class's `embed_query` calls `embed_documents`, which the subclass overrides to add the
      document prefix. Nothing errored, and the unit test passed because it mocked the method
      that added the second prefix. Fixed on 2026-10-18, with a test that checks the exact
      text sent to Ollama. Stored chunks were always right, so nothing was re-indexed.*
- [x] **Re-ingest when switching embedding models.** This became one Chroma collection per
      embedding model instead. Different models produce different vector sizes, and even
      same-size ones give meaningless matches against each other's vectors. With separate
      collections, both indexes sit in `vectorstore/` side by side, and switching without
      re-indexing finds an empty collection, which the app reports as "run `ingest.py`".
- [x] **Run the eval set against each combination** and record answer quality, speed, and
      cost. All four runs are saved in [`evals/results/`](evals/results/). Scores were 25/26
      for all OpenAI, 25/26 for `nomic-embed-text` with `gpt-4o-mini`, 25/26 for OpenAI
      embeddings with `llama3.2:3b`, and 23/26 fully local, at about 1.2 to 1.7 seconds per
      question with `gpt-4o-mini` answers and 10 to 12 with `llama3.2:3b`. Speed is wall-clock
      on one Apple M2 with 8 GB. Cost per run is under a cent with OpenAI answers and nothing
      with local ones, estimated from token counts, not billed. *The two `nomic-embed-text`
      runs were redone after the prefix fix, and both scored one lower than before it. Then
      `thread-id-config`, which had accepted only `langchain-persistence.md`, was changed to
      accept `langchain-add-memory.md` too. That file passes `{"configurable": {"thread_id":
      "1"}}` to a graph in almost every example, nomic fetched all six chunks from it, and
      `gpt-4o-mini` answered correctly from them. Both nomic runs were redone once more, and
      nomic with `gpt-4o-mini` went back to 25/26. The fully local run still fails it, because
      its answer leaves out the key.*
- [x] **Put the comparison table in the README.**

Expectation going in: Ollama is free and works offline, but slower and weaker at obeying the
"answer only from context" instruction. The eval set's eight refusal questions are there to
measure exactly that. *What happened: free and slower, both confirmed (roughly seven to nine
times slower per answer, on one Apple M2 with 8 GB, wall-clock including network and model
loading). Offline use was not tested. The grounding expectation was not borne out on this set,
because `llama3.2:3b` declined all eight. Its weaknesses showed up elsewhere, in the fully
local run. It answered `thread-id-config` but left out the `configurable` key, although all six
chunks it was given contained it. And it said "I don't know." to `grade-documents`, which the
documents answer and which `gpt-4o-mini` answered from the same six chunks. Neither failure was
an answer from outside the context.*

## Phase 4: Stretch

- [x] **Conversation memory.** Follow-up questions are rewritten into standalone ones before
      retrieval, with one extra `gpt-4o-mini` call, because retrieval happens before the
      model sees anything and "the second one" is useless as a search string. History is
      capped at six messages. *Bug found along the way: sending the rewritten question to
      the retriever but the raw one to the answering model let the two resolve the same
      pronoun differently. Right documents fetched, wrong question answered, no error
      anywhere. Both now get the rewritten question.* History lives in the Streamlit
      session, so it resets on page refresh.
- [ ] **Try searching with a drafted answer (HyDE).** Ask the model for a short draft answer
      and search with the draft instead of the question, because a draft uses the vocabulary
      of an answer. One probe on `long-term-memory-store` (2026-10-18): `gpt-4o-mini`'s draft
      invented a class that does not exist, yet searching with it moved the real answer chunk
      from 10th to 3rd with OpenAI embeddings and from 31st to 6th with `nomic-embed-text`. The
      draft is only used to search; the answer still comes from the retrieved chunks. Before
      keeping it: one extra model call per question, and its effect on the other questions,
      refusals especially, is unmeasured. Make it a setting and run the full eval.
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
- **A failed model call leaves a question without an answer.** Whether OpenAI returns an
  error or a local Ollama model dies mid-answer, which happened under memory pressure on an
  8 GB Mac, the app shows a traceback, and the unanswered question stays in the history sent
  with the next one.
- **The Sources panel renders chunks as Markdown and cuts them at 800 characters**, so
  headings appear oversized and long passages are shortened.
- **`long-term-memory-store` fails in every setup, because of how it is worded.** Investigated
  on 2026-10-18, retrieval only. The question describes the need ("persist across different
  conversation threads"), but the documents file the answer under the solution's name,
  "long-term memory". The chunk that answers it (`## Add long-term memory`, with
  `InMemoryStore` and `compile(store=store)`) ranked 10th with OpenAI embeddings and 31st with
  `nomic-embed-text`, so it was never among the six sent to the model. Another chunk that
  answers it, the Quickstart in `langchain-persistence.md`, ranked 166th and 35th. These ranks
  come from a one-off retrieval check with a throwaway script, not one saved in the repo. The
  chunks that won talk about threads and conversations, which in these documents is short-term
  memory, plus a page footer saying "use stores to persist durable data across threads". That
  is where the vague answers come from. Asked as "How do I add long-term memory to a LangGraph
  graph?", the same chunk ranked 2nd and 1st. Nothing in the code, the grader, or the models is
  broken: plain similarity search works when you already use the documents' words. The question
  stays as written, because rewording it to match would hide a real weakness. Searching with a
  drafted answer is the experiment to try next, in Phase 4.
- **`llama3.2:3b` sometimes repeats the prompt's formatting to the user.** One answer opened
  "According to <context> [2] Source: data/langchain-short-term-memory.md </context>".
  No `gpt-4o-mini` answer in the saved runs does this.

## Model choice

Staying on `gpt-4o-mini` as the default, with `llama3.2:3b` as the local option. As of
2026-10-12, OpenAI's deprecations page lists no shutdown date for `gpt-4o-mini`, and it
accepts `temperature=0`, which the app relies on for repeatable answers.

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
- The `nomic-embed-text` prefixes (`search_document: ` and `search_query: `) were added
  because its model card requires them. Until 2026-10-18 questions wrongly got both
  prefixes. Fixing that changed which chunks made the top 6 for 18 of the 26 eval questions,
  measured on retrieval alone, and lowered both `nomic-embed-text` eval scores by one. Running
  with no prefix at all has not been tested.
- Ollama's behaviour on a prompt longer than its context window was avoided, not tested.
  Ollama defaults to 4,096 tokens on machines like this one, and `models.py` sets 8,192. The
  longest prompt measured (six long history messages plus six chunks) was 2,107 tokens, in
  a one-off check with a throwaway script, not one saved in the repo.
- The speed figures come from one Apple M2 with 8 GB, one run per combination. They are
  wall-clock from `evals/run.py`, so they include network round trips for OpenAI and model
  loading for Ollama. On this machine the local answer runs only completed with Chrome and
  VS Code closed.
- Code review, 2026-08-05: no secrets in `data/`, and `.env` is gitignored. The documents are
  full of `{"configurable": {"thread_id": "1"}}` samples and `ChatPromptTemplate` uses
  f-string syntax, so a runtime `KeyError` looked likely. Tested rather than assumed:
  substitution is single-pass and braces in values stay literal. Not a bug.
