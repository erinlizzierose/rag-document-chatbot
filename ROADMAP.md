# Roadmap

Goals for this project: understand how retrieval augmented generation actually works, and
end up with a small, presentable repo.

---

## Current state, 2026-08-31

**The app works.** Five commits on `main`, index healthy at 240 vectors across six documents.
Run it with `streamlit run app.py`.

What it does today:

- Answers questions from the documents in `data/`, showing the exact passages used
- Declines instead of inventing when the documents don't cover something
- Handles follow-up questions ("and what about the other one?") by rewriting them into
  standalone questions before searching

**Not committed:**

- The five `data/langchain-*.md` files are untracked. The index is built from them, and git
  does not know they exist.
- `stash@{0}` holds the Phase 2 housekeeping: MIT `LICENSE`, `data/SOURCES.md` attribution,
  pinned `requirements.txt`, and deletion of the stale `sample_notes.md`. Recover with
  `git stash pop`.

**Known gaps in the docs themselves:**

- `docs/screenshot.png` does not exist, so the README renders a broken image.
- The README does not yet mention that conversation memory is built.

**Suggested next step:** Phase 3, starting with the eval set. Judging answer quality by feel
has already stopped scaling.

---

## Phase 0 — Make it real (complete 2026-08-07)

- [x] Cleared a stale empty `vectorstore/`
- [x] **Added real documents to `data/`.** `sample_notes.md` alone was 927 characters, one
      chunk, so with `k=4` the retriever handed over the whole document every time and
      retrieval was a no-op. Added five LangChain documentation pages.
- [x] Added a real `OPENAI_API_KEY` to `.env`
- [x] `python ingest.py` → 242 chunks from 6 files in 7.5s, verified in the store
- [x] Asked a question the documents answer. Retrieval selected chunks from only
      `langchain-persistence.md` and `langchain-add-memory.md` out of six files.
- [x] Asked one they do not answer ("capital of France") and it declined.
      *Retrieval still returned 4 chunks, because similarity search has no concept of "no
      match." The grounding instruction is what turns that into an honest refusal.*

## Phase 1 — Correctness (complete 2026-08-07, except the screenshot)

- [x] **Fix duplicate ingest.** Now constructs the store, calls `reset_collection()`, then
      `add_documents()`. `Chroma.from_documents` generated a fresh UUID per chunk against a
      get-or-create collection, so every run appended and re-billed the embeddings.
      *Verified: a second ingest left 240 vectors, not 480.*
- [x] **Name the embedding model explicitly in both files.** Both set
      `EMBEDDING_MODEL = "text-embedding-3-small"` with paired comments. The old default,
      `text-embedding-ada-002`, is also 1536 dimensions, so a mismatch raises nothing and
      simply returns bad matches.
- [x] **Guard the empty index.** Branches on an empty `retriever.invoke()` result and returns
      before constructing the model, so no API call is paid for.
- [x] **Friendlier missing-key error.** `main()` checks `OPENAI_API_KEY` up front instead of
      failing as a raw traceback on the first question. `load_dotenv()` moved to module scope
      so the app is importable without Streamlit, which Phase 3's eval scripts need.
- [x] **Move retrieved context out of the system message.** Context sits in the human turn
      inside `<context>` tags; the system prompt says tagged text is untrusted document
      content, never instructions.
- [x] **Strip agent-directed boilerplate during ingest.** `strip_leading_blockquote()` drops
      only the contiguous `>` run at the top, so real blockquotes further down survive.
      *Measured: on "summarize everything you know about memory," 2 of 4 retrieved chunks used
      to be boilerplate. Now 0 of 4.*
- [x] **Rewrite the README.** Leads with what the project does, documents that a fresh clone
      has no `vectorstore/`, states real costs, and includes a "What I learned" section.
- [x] Removed the duplicated source-label formatting; both call sites use `describe_source()`.
      *(Pulled forward from Phase 2.)*
- [ ] **Screenshot at `docs/screenshot.png`, Sources panel expanded.** Only Erin can take this.
      The README already links it and renders broken until it exists.

## Phase 2 — Polish (written, currently stashed)

All five are done and sitting in `stash@{0}`. Erin set them aside on 2026-08-11 to prioritise
logic work. `git stash pop` restores them. Marked `[~]` rather than `[x]` because they are not
in the working tree.

- [~] Pin `requirements.txt` by hand. Not `pip freeze`: the venv holds langchain-community,
      langchain-classic, and langchain-protocol, which are never imported.
- [~] Drop bare `langchain` from requirements. Nothing imports it; in LangChain 1.x it is a
      thin namespace and every piece used here lives in its own package.
- [~] Add an MIT `LICENSE`.
- [~] **Attribution for the documents in `data/`.** They are LangChain documentation pages,
      about 180 KB. Verified MIT licensed via the `langchain-ai/docs` repo, so redistribution
      with attribution is fine. `data/SOURCES.md` lists each URL, the fetch date (2026-08-04),
      the upstream license, and a staleness warning.
      **Settle this before the repo goes public.**
- [~] Delete `data/sample_notes.md`. It still claims to exist "so the chatbot has something to
      answer questions about before you add your own notes or PDF," which is no longer true.

## Phase 3 — Run it locally too, and compare

The item that makes this more than a tutorial clone. Same pipeline, two backends, measured.

- [ ] **Write the eval set first.** Twenty or so question/expected-answer pairs. Without it,
      every comparison below is a vibe. It also gives the retrieval-quality questions raised
      during Phase 1 something real to measure against.
- [ ] `pip install langchain-ollama` (1.1.0, compatible with the installed langchain-core),
      install Ollama, pull a chat model and `nomic-embed-text`
- [ ] Make the backend configurable. Keep the **embedding model and chat model as separate
      settings**, not one switch. Local embeddings with an API chat model is a reasonable
      combination and shows which half of RAG drives answer quality.
- [ ] Re-ingest when switching embedding backends. Different models produce different vector
      sizes and the old database becomes unusable.
- [ ] Run the eval set through both. Record answer quality, speed, and cost.
- [ ] Put the comparison table in the README.

Ollama is free and works offline, but slower and weaker at obeying the "answer only from
context" instruction.

## Phase 4 — Stretch

- [x] **Conversation memory.** Done 2026-08-11, and done properly rather than the cheap version
      originally planned. `build_search_query()` makes one extra `gpt-4o-mini` call to rewrite a
      follow-up into a standalone question *before* retrieval, because retrieval happens before
      the model sees anything and "the second one" is useless as a search string. History is
      capped at `MAX_HISTORY_MESSAGES = 6`.
      *Bug found and fixed during implementation: sending the rewritten question to the
      retriever but the raw one to the answering model made the two components resolve the same
      pronoun differently. Right documents fetched, wrong question answered, and no error
      anywhere. Both now receive the rewritten question.*
      *Remaining limitation: history lives in `st.session_state`, so it resets on page refresh.*
- [ ] **Document upload in the app.** Currently adding a document means copying a file and
      running a terminal command. This is the gap between "works for me" and "usable by others."
- [ ] Compare chunk sizes, now meaningful since `data/` holds real documents.
- [ ] Support multiple collections.

Not recommended: the framework-native memory route. `langchain.memory` was removed in
LangChain 1.0 and `RunnableWithMessageHistory` warns as deprecated since langchain-core 1.3.3.
The official path is `create_agent` plus a LangGraph checkpointer, which is a large detour.
The hand-rolled rewriting above teaches more for this project's purposes.

---

## Model choice

Staying on `gpt-4o-mini`. No announced retirement, and it preserves `temperature=0`.
Note for later: `langchain-openai` silently discards `temperature` for models whose name starts
with `gpt-5`, unless the name contains `chat` or reasoning effort is `none`. Switching to a
`gpt-5.x` model would quietly void the determinism this app relies on.

## Verification notes

OpenAI pricing and retirement dates were checked once on 2026-08-04 and are single-source.
The PDF branch of `load_documents` has still never executed; `data/` has only ever held
Markdown. The claim that mismatched 1536-dimension embedding models fail silently is inferred
from Chroma validating vector width only; it was not reproduced with two live models.

Code review, 2026-08-05. No secrets in `data/` or `.env`; `.env` correctly gitignored. Tested
rather than assumed: the docs are full of `{"configurable": {"thread_id": "1"}}` code samples
and `ChatPromptTemplate` uses f-string syntax, so a runtime `KeyError` looked likely.
Substitution is single-pass and braces in values stay literal. Not a bug.
