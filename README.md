# Document Chatbot

Ask questions about your own documents and get answers grounded in them, with the exact
source passages shown alongside every response.

Built to learn how retrieval augmented generation actually works, rather than to wrap an
API and call it a chatbot.

<!-- Replace with a screenshot of the app with the Sources panel expanded. -->
![Screenshot](docs/screenshot.png)

## What it does

Point it at a folder of Markdown, text, or PDF files. It splits them into chunks, converts
each chunk into a vector, and stores them locally. When you ask a question, it finds the
handful of chunks most similar to your question and asks a language model to answer using
only those.

If the answer isn't in your documents, it says so instead of inventing one.

## How it works

1. `ingest.py` reads every supported file in `data/`
2. Splits them into overlapping ~1000 character chunks
3. Converts each chunk to an embedding, a vector capturing its meaning
4. Stores chunks and vectors in a local Chroma database
5. `app.py` embeds your question the same way and retrieves the 4 nearest chunks
6. Those chunks go to the model inside a prompt that forbids answering from anything else

The Sources panel shows you exactly which chunks were used, so you can check whether the
answer is actually supported by them.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then add your OpenAI API key
```

Put some documents in `data/`, then:

```bash
python ingest.py          # build the index
streamlit run app.py      # start the app
```

`vectorstore/` is gitignored, so **a fresh clone has no index**. Running `ingest.py` is a
required step, not an optional one. The app will tell you if you skip it.

## Cost

Indexing a 100-page PDF costs roughly a tenth of a cent. Each question costs about
$0.0003, so a dollar buys around 3,000 questions. Uses `text-embedding-3-small` for
embeddings and `gpt-4o-mini` for answers.

## What I learned building this

**Retrieval and grounding are separate mechanisms, and only one of them is search.**
Ask something the documents don't cover and it still retrieves 4 chunks, because similarity
search has no concept of "no match." It returns the least-bad options every time. The
refusal comes entirely from the prompt, not the retrieval.

**This architecture cannot summarize a corpus.** Nothing in the pipeline ever reads all the
chunks. Ask it to summarize everything and it will confidently summarize the 4 chunks it
happened to retrieve, which reads exactly like a complete answer. Asking a specific question
retrieves specific content; asking a vague one retrieves vague content.

**Junk in the corpus competes for retrieval slots.** The documentation pages here opened
with a boilerplate header addressed to an LLM. It was the only text in the corpus discussing
documentation in the abstract, so it beat real content on precisely the broad questions where
a wasted slot hurts most, taking 2 of 4 slots. Stripping it changed the answers noticeably.

**Two embedding models with the same vector size fail silently.** The old default,
`text-embedding-ada-002`, and `text-embedding-3-small` are both 1536 dimensions. Embedding
documents with one and questions with the other raises no error at all and simply returns
wrong matches. Both files pin the model explicitly, with comments explaining why they must
agree.

## Project layout

```text
data/              Documents to index
vectorstore/       Local Chroma database, created by ingest.py, gitignored
app.py             Streamlit chat interface
ingest.py          Document loading and indexing
ROADMAP.md         Planned work and known issues
```

## Adding your own documents

Drop `.md`, `.txt`, or `.pdf` files into `data/` and re-run `python ingest.py`. Re-running is
safe and repeatable; it clears the collection before rebuilding.

PDFs need a real text layer. Scanned or photographed pages have none, so the text extracts as
empty and you get an index that silently knows nothing.

## Next steps

See [ROADMAP.md](ROADMAP.md). The most interesting one is running the same pipeline against a
local Ollama model and comparing answer quality, speed, and cost against the API.
