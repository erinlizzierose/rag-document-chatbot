# Document Chatbot

A small chatbot that answers questions about a folder of documents and shows the passages each
answer came from. I built it to learn how retrieval augmented generation (RAG) works, so the
code is short and this README records what I found along the way.

![Screenshot](docs/screenshot.png)

## What it does

You put Markdown, text or PDF files in `data/`. The app splits them into chunks, turns each
chunk into a vector, and stores the vectors on your machine. When you ask a question, it finds
the six chunks closest to your question and asks a language model to answer from those alone.

If the documents don't cover the question, it says it doesn't know. Follow-up questions such as
"and what about the second one?" also work, because the app rewrites them into standalone
questions before it searches.

Five pages of LangChain documentation come with the repo, so there's something to ask about
right away. [data/SOURCES.md](data/SOURCES.md) says where they came from.

## How it works

1. `ingest.py` reads every `.md`, `.txt` and `.pdf` file in `data/`.
2. It splits them into overlapping chunks of about 750 characters.
3. Each chunk becomes an embedding, a list of numbers that captures what it means.
4. The chunks and embeddings go into a Chroma database in `vectorstore/`.
5. When you ask something in `app.py`, a follow-up is first rewritten to stand on its own.
6. The question is embedded the same way, and the 6 nearest chunks are retrieved.
7. Those chunks go to the model in a prompt that tells it to answer only from them.

Steps 3 and 6 use an embedding model, and steps 5 and 7 use a chat model. Each one can be OpenAI
or a local model run by Ollama, set in `.env`. Both default to OpenAI, and `models.py` builds
whichever you choose.

To read the code, start at `answer_question()` in `app.py`, which is about 30 lines. Every
answer has a Sources panel listing the chunks it was given (the first 800 characters of each),
so you can check the answer against them.

## Setup

You need Python 3.10 or newer, and for the default setup an OpenAI API key. On a Mac the
built-in `python3` may be 3.9, which is too old. Check with `python3 --version` and install a
newer one from python.org or Homebrew if you need to.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then put your OpenAI key after OPENAI_API_KEY=
python ingest.py          # build the index from data/
streamlit run app.py      # start the app
```

A fresh clone has no index, because `vectorstore/` is gitignored, so `ingest.py` has to run
before the app has anything to search. If you skip it, the app tells you.

### Running it locally with Ollama

[Ollama](https://ollama.com) runs models on your own computer, with no API key and no cost per
question. Install it as an app from ollama.com, or with Homebrew on a Mac:

```bash
brew install ollama
ollama serve                  # keep this running in its own terminal tab
```

If you'd rather have it run in the background, use `brew services start ollama` instead of
`ollama serve`. It will then also start at every login, until you run
`brew services stop ollama`.

Then download the two models (2.0 GB and 274 MB):

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Set the backends in `.env`. They're independent, so you can mix them, for example local
embeddings with OpenAI answers:

```text
EMBEDDING_BACKEND=ollama
CHAT_BACKEND=ollama
```

Run `python ingest.py` again, because each embedding model has its own index, and restart the
app if it was running. With both set to `ollama` you don't need an OpenAI key.

8 GB of RAM is enough, but only just. On my 8 GB M2 the local runs crashed while Chrome and
VS Code were open, and worked once I quit them.

## Evaluating it

`evals/questions.yaml` has 26 questions. The documents answer 18 of them, and the app should
decline the other 8. Three of those 8 are near misses: they ask about something the documents
discuss but never answer. `evals/run.py` asks every question and checks two things separately:
whether the answer contains the key terms, and whether retrieval found the right file.
Retrieval tends to slip before the answers do, so the second check warns you earlier.

```bash
python evals/run.py               # every question
python evals/run.py trim          # only questions whose id contains "trim"
RETRIEVAL_K=4 python evals/run.py # try a different number of retrieved chunks
```

With OpenAI for both models it scores 25 of 26 (2026-10-18). The one miss,
`long-term-memory-store`, is explained under What I learned below.

I used the eval to choose two settings. Retrieving 6 chunks scored better than 4, and 8 or 10
added nothing. Chunks of 1000 characters buried one answer at rank 15, and 750 brought it to
first. Those sweeps ran on an older, smaller version of the question set.

The unit tests need no API key and no Ollama:

```bash
python -m unittest discover -s tests -v
```

## OpenAI against a local model

I ran the eval with every combination of OpenAI and Ollama on 2026-10-18:

| Embeddings | Answers | Score (of 26) | Answerable (of 18) | Refusals (of 8) | Seconds per question | Cost per run |
|---|---|---|---|---|---|---|
| `text-embedding-3-small` | `gpt-4o-mini` | 25 | 17 | 8 | 1.7 | under 1 cent |
| `nomic-embed-text` | `gpt-4o-mini` | 25 | 17 | 8 | 1.2 | under 1 cent |
| `text-embedding-3-small` | `llama3.2:3b` | 25 | 17 | 8 | 11.3 | close to zero |
| `nomic-embed-text` | `llama3.2:3b` | 23 | 15 | 8 | 11.1 | $0 |

The times come from one 8 GB Apple M2 and include network time for OpenAI and model loading for
Ollama. Costs are estimated from token counts. The output of each run is in
[`evals/results/`](evals/results/).

Every combination declined all 8 questions it should have, including the near misses. I
expected the small local model to make things up more often, and it didn't. Where it fell short
was detail. In the fully local run it left out the `configurable` key from its answer to
`thread-id-config`, although every chunk it was given showed that key, and it said "I don't
know" to a question its chunks did answer. The same model got both right when OpenAI's
embeddings picked the chunks. It was also about seven to nine times slower.

With 26 questions and one run shown per row, a gap of one or two points is a hint more than a
result. The [roadmap](ROADMAP.md) has the full story, including a bug in the local embedding
setup and an eval check that was too strict, which both moved the local scores along the way.

## Cost

Indexing the bundled 177 KB of documents with `text-embedding-3-small` costs about a tenth of a
cent. One question to `gpt-4o-mini` costs about $0.0002: I measured one at 1,035 tokens in and
111 out. That's roughly 4,500 questions per dollar. A follow-up question adds one smaller call to
rewrite it. These figures come from token counts at OpenAI's listed prices (`gpt-4o-mini`
checked 2026-10-19), and I haven't compared them with a bill. The local models cost nothing per
question.

## What I learned building this

**Search always returns something.** Ask about a topic the documents don't cover and retrieval
still hands back 6 chunks, because similarity search has no idea of "no match". It returns the
closest ones it has. The refusal comes from the prompt, which tells the model to answer only
from those chunks.

**It can't summarize the whole corpus.** Nothing in the pipeline reads every chunk. Ask for a
summary of everything and you get a confident summary of the six chunks that happened to come
back, and it reads like a complete answer.

**A chunk's embedding describes the whole chunk.** One answer was a short section inside a
1000-character chunk that was mostly import examples, so the chunk's vector mostly described
the imports. A question using nearly the documentation's own words still ranked it 15th.
Smaller chunks fixed it, but at 300 characters two other answers were split across chunks and
broke.

**Search matches your words, and the documents may use different ones.** Asked how to store
information "across different conversation threads", every setup gave a vague answer. The
documents call that "long-term memory", and the words "conversation" and "thread" mostly appear
on the short-term memory pages, so those pages won. With OpenAI embeddings the right chunk came
10th. Asked about "long-term memory" directly, it came 2nd. I left the question as it is so the
eval keeps showing the weakness.

**Some mistakes raise no error at all.** If documents are indexed with one embedding model and
questions searched with another of the same vector size, you just get wrong matches. That's why
`models.py` is the one place that names the models, and each model gets its own index. The local
embedding model also needs a label in front of each text, and my code was quietly adding the
wrong one to every question. Fixing it changed which chunks came back for 18 of the 26
questions. The unit test passed the whole time, because it mocked the method that added the
extra label.

**Follow-up questions break the search too.** "How do I add the second one?" shares no words
with the answer, and the search happens before the model sees anything. So the app rewrites the
question first and sends the rewritten version to both the search and the model. When only the
search got it, the two could take "the second one" to mean different things.

**Documents can break your prompt.** The prompt puts retrieved text inside `<context>` tags and
says everything in there is untrusted. The bundled docs contain their own `</context>` tags in
example prompts, which would close the block early. The app now escapes those tags first. That
makes the trick harder, but it doesn't make it impossible.

**An eval check can pass by accident.** One question required the word "delete", which the docs
use only in a code comment. Their actual advice is to "prune". The check passed for weeks, then
failed a correct answer after a chunking change altered what the model saw.

## Project layout

```text
app.py             Streamlit chat interface and the question-answering logic
ingest.py          Document loading and indexing
models.py          Which models embed text and write answers, OpenAI or Ollama
data/              Documents to index, plus SOURCES.md (attribution, not indexed)
evals/             Question set and the script that scores the app against it
evals/results/     Saved eval runs, one per combination of models
tests/             Offline unit tests for the parts that don't call a model
docs/              Screenshot for this README
vectorstore/       Local Chroma database, created by ingest.py, gitignored
ROADMAP.md         What's been done, what's next, and known issues
```

## Adding your own documents

Drop `.md`, `.txt` or `.pdf` files into `data/` and run `python ingest.py` again. It clears the
old index first, so re-running is safe. If you replace the bundled documents, the eval questions
won't apply any more, since they're about those documents.

PDFs need a real text layer. Scanned pages have none, so they're left out of the index without a
warning. If nothing in `data/` has any text, `ingest.py` stops and leaves the existing index
alone. PDF loading is also the least tested part, since the bundled documents are all Markdown.

## Limitations

- Conversation history lives in the Streamlit session and resets when you refresh the page.
- There's no upload button. Adding documents means copying files and running `ingest.py`.
- The local-model numbers come from one 8 GB Apple M2.
- The bundled documents are a snapshot from 2026-08-04 and will drift from the live LangChain
  docs.

## Next steps

See [ROADMAP.md](ROADMAP.md). Next on my list is searching with a drafted answer instead of the
question, a technique called HyDE. One quick test suggests it would fix `long-term-memory-store`.

## License

The code is MIT licensed; see [LICENSE](LICENSE). The documents in `data/` are LangChain's
documentation, also MIT licensed, copyright LangChain. [data/SOURCES.md](data/SOURCES.md) lists
where each one came from and includes their license.
