from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

import models


DATA_DIR = Path("data")
VECTORSTORE_DIR = Path("vectorstore")
SUPPORTED_EXTENSIONS = {".md", ".pdf", ".txt"}

# Files that live in data/ but are about the corpus rather than part of it. SOURCES.md is
# attribution, and it belongs beside the documents it attributes. Indexed, though, it is two
# chunks of citation table and staleness warning: the only text in the corpus discussing
# documentation in the abstract, which is exactly what wins slots on vague questions. Same
# problem strip_leading_blockquote() exists to solve, arriving by a different route.
EXCLUDED_FILENAMES = {"SOURCES.md"}

# A chunk is embedded as one vector, so a short section that shares its chunk with unrelated
# text gets an embedding that mostly describes the unrelated text. At 1000 characters the
# migration guide's short "return type" section shared a chunk with import examples and
# a Python 3.9 note, and ranked 15th for a question asking about it almost word for word.
#
# Swept on 2026-10-14, retrieval only: 1000 left 17/18 answerable questions with the right
# text in the top 6 chunks; 750 and 500 got 18/18; 300 dropped to 16/18 by splitting other
# answers across chunks. 750 is the largest size that fixes it, which keeps the most context
# per retrieved chunk. Changing these means re-running ingest.py.
CHUNK_SIZE = 750
CHUNK_OVERLAP = 150


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

    # Checked before anything is read or deleted. ingest.py only embeds, so the chat backend
    # does not need to be set up for it.
    problem = models.setup_problem(include_chat=False)
    if problem:
        raise SystemExit(problem)

    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    # Checked before reset_collection(), which deletes the existing index. Scanned PDFs and
    # empty files load without error but contain no text, so the splitter returns nothing.
    # Without this, the old index was wiped and then Chroma failed on an empty add.
    if not chunks:
        raise SystemExit(
            f"No text could be extracted from the files in {DATA_DIR}/. Scanned PDFs have no "
            "text layer, and empty files have nothing to index. The existing index was left "
            "untouched."
        )

    vectorstore = Chroma(
        collection_name=models.collection_name(),
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=models.make_embeddings(),
    )

    # Chroma.from_documents() would append. It generates a fresh UUID for every chunk
    # against a get-or-create collection, so running this script twice stores two copies
    # of everything, charges you twice, and degrades retrieval by returning duplicates.
    # Clearing first makes re-indexing safe to repeat.
    vectorstore.reset_collection()
    vectorstore.add_documents(chunks)

    print(f"Indexed {len(chunks)} chunks from {len(documents)} document pages/files.")
    print(f"Embedded with {models.embedding_model_name()}, saved to {VECTORSTORE_DIR}/")


if __name__ == "__main__":
    main()
