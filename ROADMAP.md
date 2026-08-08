# Roadmap

Goals for this project: understand how retrieval augmented generation actually works, and
end up with a small, presentable repo.

Status as of 2026-08-04: the pipeline is written but has never completed a run end to end.

---

## Phase 0 — Make it real

The project can't teach anything until it runs once on a document worth searching.

- [x] `rm -rf vectorstore/` to clear a stale empty database
- [x] **Add a longer document to `data/`.** This is the highest-value item in the whole roadmap.
      `sample_notes.md` is 927 characters, which fits in a single chunk. With one chunk and
      `k=4`, the retriever hands over the entire document every time. Retrieval is a no-op
      until there is enough text to choose between.
      *Done 2026-08-05: added five LangChain documentation pages, 242 chunks total.*
- [x] `cp .env.example .env` and add a real `OPENAI_API_KEY`
- [x] `python ingest.py` and confirm the printed chunk count is well above 1
      *Done 2026-08-07: 242 chunks from 6 files in 7.5s, 242 vectors verified in the store.*
- [x] `streamlit run app.py`, ask a question the document answers
      *Asked about LangGraph checkpointers and `thread_id`. Retrieval selected chunks from only
      `langchain-persistence.md` and `langchain-add-memory.md` out of six files. Correct answer,
      with a citation.*
- [x] Ask a question the document does **not** answer, confirm it declines
      *Asked for the capital of France. Answered "I do not know." Note that retrieval still
      returned 4 chunks, because similarity search has no concept of "no match." The grounding
      instruction in the system prompt is what turns that into an honest refusal.*

**Phase 0 complete 2026-08-07.** The pipeline runs end to end for the first time.

## Phase 1 — Correctness

- [x] **Fix duplicate ingest.** `ingest.py` now constructs the store, calls
      `reset_collection()`, then `add_documents()`. `Chroma.from_documents` generated a fresh
      UUID per chunk against a get-or-create collection, so every run appended instead of
      replacing and re-billed the embeddings.
      *Verified 2026-08-07: a second ingest into an existing store left 240 vectors, not 480.*
- [x] **Name the embedding model explicitly in both files.** Both now set
      `EMBEDDING_MODEL = "text-embedding-3-small"` with a comment saying they must match.
      `OpenAIEmbeddings()` with no argument had been silently resolving to
      `text-embedding-ada-002`. Both models are 1536 dimensions, so a mismatch raises nothing
      and just returns bad matches.
- [x] **Guard the empty index** in `app.py`. Branches on an empty result from
      `retriever.invoke()` and returns before constructing the model, so no API call is paid
      for. *Verified with a stubbed retriever: returns the "run python ingest.py" message.*
- [x] **Rewrite the README.** Leads with what the project does, documents that a fresh clone
      has no `vectorstore/` so `ingest.py` is mandatory, and states real costs. Includes a
      "What I learned" section covering the four non-obvious things this project surfaced:
      retrieval and grounding being separate mechanisms, the impossibility of corpus
      summarization, junk competing for retrieval slots, and same-dimension embedding models
      failing silently.
      **Still needed: a screenshot at `docs/screenshot.png`, with the Sources panel expanded.**
      The image placeholder is already in the README and will render broken until it exists.

**Phase 1 complete 2026-08-07**, apart from the screenshot, which only Erin can take.
- [x] **Friendlier missing-key error.** `main()` checks `OPENAI_API_KEY` up front and shows a
      Streamlit error telling you to fill in `.env`, instead of a raw traceback on the first
      question. `load_dotenv()` also moved to module scope so the app is importable by the
      Phase 3 eval scripts without launching Streamlit.
- [x] **Move retrieved context out of the system message.** Context now sits in the human turn
      inside `<context>` tags, and the system prompt says the tagged block is untrusted
      document text to be described, never instructions to follow.
      *Verified: document text appears in the human message and not in the system message.*
- [x] **Strip agent-directed boilerplate during ingest.** `strip_leading_blockquote()` drops a
      leading `>` block. Only the contiguous run at the top, so real blockquote content further
      down survives.
      *Measured impact: on "summarize everything you know about memory," 2 of the 4 retrieved
      chunks used to be this boilerplate. Now all 4 are real content, including the
      checkpointer-vs-store comparison table.*

Also pulled forward from Phase 2 while rewriting `app.py`:

- [x] Removed the duplicated source-label formatting. Both call sites now use
      `describe_source()`.

## Phase 2 — Polish

- [ ] Pin `requirements.txt` by hand. Do not `pip freeze`: the venv contains undeclared
      transitive packages that would make the repo look worse.
- [ ] Drop bare `langchain` from requirements. Nothing imports it.
- [ ] Add an MIT `LICENSE`
- [ ] Remove the duplicated source-label formatting in `app.py` (two copies)
- [ ] **Sort out attribution for the documents in `data/`.** They are LangChain documentation
      pages, about 180 KB, and `.gitignore` does not exclude them, so they will be published.
      Preferred fix: gitignore `data/*.md` apart from `sample_notes.md` and ship a small
      `fetch_docs.py` instead. Smaller repo, no redistribution question, and the fetch script is
      a better portfolio artifact than a pile of copied Markdown. Minimum fix: a `data/SOURCES.md`
      listing each URL, the fetch date, and the upstream license.
      **Do this before the repo goes public.**
- [ ] Rewrite or delete `data/sample_notes.md`. It still claims to exist "so the chatbot has
      something to answer questions about before you add your own notes or PDF," which is no
      longer true. As the only single-chunk file it will occasionally surface as a source
      describing the project's own scaffolding.

## Phase 3 — Run it locally too, and compare

This is what makes the project interesting rather than another tutorial clone. Same pipeline,
two different backends, measured against each other on real questions.

- [ ] `pip install langchain-ollama`, install Ollama, pull a chat model and `nomic-embed-text`
- [ ] Make the backend configurable. Keep the **embedding model and the chat model as separate
      settings**, not one switch. Local embeddings with an API chat model is a reasonable
      combination, and mixing them deliberately shows which half of RAG drives answer quality.
- [ ] Delete `vectorstore/` and re-ingest when switching embedding backends. Different models
      produce different vector sizes and the old database is unusable.
- [ ] Write a small set of question / expected-answer pairs
- [ ] Run that set through both backends. Record answer quality, speed, and cost.
- [ ] Put the comparison table in the README.

Rough cost of the OpenAI side: about a tenth of a cent to index a 100-page PDF, and roughly
3,000 questions per dollar with `gpt-4o-mini`. The real cost is OpenAI's prepaid minimum,
not usage. Ollama is free but slower, and weaker at obeying the "answer only from context"
instruction.

## Phase 4 — Stretch

- [ ] **Conversation memory, cheap version.** `st.session_state.messages` already holds the full
      history and is simply never passed to the model, so follow-up questions don't work today.
      Feeding the last few turns into the prompt is about ten lines and teaches what "memory"
      actually means.
- [ ] Compare chunk sizes. Only meaningful once `data/` holds a document longer than one chunk.
- [ ] Document upload directly in the app
- [ ] Support multiple collections

Not recommended yet: the framework-native memory route. `langchain.memory` was removed in
LangChain 1.0, and `RunnableWithMessageHistory` now warns as deprecated. The official path is
`create_agent` plus a LangGraph checkpointer, which is a large detour from the goals above.

---

## Model choice

Staying on `gpt-4o-mini`. No announced retirement, and it preserves `temperature=0`.
Note for later: `langchain-openai` silently discards `temperature` for models whose name starts
with `gpt-5` (unless the name contains `chat` or reasoning effort is `none`). Switching to a
`gpt-5.x` model would quietly void the determinism this app relies on.

## Verification notes

OpenAI pricing and retirement dates were checked once on 2026-08-04 and are single-source.
The PDF branch of `load_documents` has never executed. The claim that mismatched 1536-dimension
embedding models fail silently is inferred from Chroma validating vector width only; it was not
reproduced with two live models.

Code review, 2026-08-05. Checked and clean: no secrets in `data/` or `.env` (scanned for OpenAI,
GitHub, AWS, and private-key patterns); `.env` correctly gitignored. Also tested rather than
assumed: the new docs are full of `{"configurable": {"thread_id": "1"}}` code samples, and
`ChatPromptTemplate` uses f-string syntax, so a `KeyError` at runtime looked likely. Substitution
is single-pass and braces in both context and question are treated as literal values. Not a bug.
