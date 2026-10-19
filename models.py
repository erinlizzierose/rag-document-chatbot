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
# 24 GiB of GPU memory. The worst prompt measured here, six retrieved chunks plus six long
# messages of history, was 2,107 tokens, about half that default. The window is set explicitly
# anyway, at twice the default, so the app never depends on what Ollama does with a prompt that
# overflows.
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
        # Not super().embed_query: it calls self.embed_documents, which would add the document
        # prefix in front of this one.
        return super().embed_documents([f"search_query: {text}"])[0]


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
        # raises a ConnectionError or ValueError saying which, before the first search rather
        # than during it.
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


def _readable(error):
    """One sentence for an error raised while creating a model.

    pydantic runs the model's validation inside its constructor and wraps whatever that raises,
    so a missing Ollama model arrives as a multi-line "1 validation error" dump. The sentence
    we want is the first entry's msg.
    """
    if hasattr(error, "errors"):
        return error.errors()[0]["msg"].removeprefix("Value error, ")
    return str(error)


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

        # Creating an Ollama model checks the server and the model, see make_embeddings(). A
        # stopped server arrives as the builtin ConnectionError, which langchain_ollama does not
        # convert, and a missing model as a ValueError.
        if backend("EMBEDDING_BACKEND") == "ollama":
            make_embeddings()
        if include_chat and backend("CHAT_BACKEND") == "ollama":
            make_chat_model()
    except (ValueError, ConnectionError) as error:
        return _readable(error)
    return None


def describe():
    """One line naming the models in use, for the app caption and the top of an eval run."""
    return f"Answers from {chat_model_name()}, search with {embedding_model_name()}"
