from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


DATA_DIR = Path("data")
VECTORSTORE_DIR = Path("vectorstore")
SUPPORTED_EXTENSIONS = {".md", ".pdf", ".txt"}

# Files that live in data/ but are about the corpus rather than part of it. SOURCES.md is
# attribution, and it belongs beside the documents it attributes. Indexed, though, it is two
# chunks of citation table and staleness warning: the only text in the corpus discussing
# documentation in the abstract, which is exactly what wins slots on vague questions. Same
# problem strip_leading_blockquote() exists to solve, arriving by a different route.
EXCLUDED_FILENAMES = {"SOURCES.md"}

# Must match EMBEDDING_MODEL in app.py. Questions are embedded by the app and compared
# against chunks embedded here, so two different models produce meaningless comparisons.
# The old default (text-embedding-ada-002) has the same 1536 dimensions as this one, so a
# mismatch raises no error at all. It just silently returns bad matches.
EMBEDDING_MODEL = "text-embedding-3-small"


def strip_leading_blockquote(text):
    """Drop a blockquote block at the very top of a document.

    Pages fetched from documentation sites often open with a note addressed to an LLM,
    telling it where to find an index. It is not content, and because it is the only text
    in the corpus that talks about documentation in the abstract, it beats real content on
    exactly the vague questions where you can least afford to waste a retrieval slot.
    """
    lines = text.splitlines()
    if not lines or not lines[0].startswith(">"):
        return text

    index = 0
    while index < len(lines) and lines[index].startswith(">"):
        index += 1

    return "\n".join(lines[index:]).lstrip()


def load_documents():
    files = [
        path
        for path in DATA_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and path.name not in EXCLUDED_FILENAMES
    ]

    if not files:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise FileNotFoundError(
            f"No documents found in {DATA_DIR}/. Add a file with one of: {supported}"
        )

    documents = []
    for path in files:
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(path)
            for page_index, page in enumerate(reader.pages):
                documents.append(
                    Document(
                        page_content=page.extract_text() or "",
                        metadata={"source": str(path), "page": page_index},
                    )
                )
        else:
            documents.append(
                Document(
                    page_content=strip_leading_blockquote(
                        path.read_text(encoding="utf-8")
                    ),
                    metadata={"source": str(path)},
                )
            )

    return documents


def main():
    load_dotenv()

    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)

    vectorstore = Chroma(
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )

    # Chroma.from_documents() would append. It generates a fresh UUID for every chunk
    # against a get-or-create collection, so running this script twice stores two copies
    # of everything, charges you twice, and degrades retrieval by returning duplicates.
    # Clearing first makes re-indexing safe to repeat.
    vectorstore.reset_collection()
    vectorstore.add_documents(chunks)

    print(f"Indexed {len(chunks)} chunks from {len(documents)} document pages/files.")
    print(f"Vector store saved to {VECTORSTORE_DIR}/")


if __name__ == "__main__":
    main()
