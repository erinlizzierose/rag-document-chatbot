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


def load_documents():
    files = [
        path
        for path in DATA_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
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
                    page_content=path.read_text(encoding="utf-8"),
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

    Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=str(VECTORSTORE_DIR),
    )

    print(f"Indexed {len(chunks)} chunks from {len(documents)} document pages/files.")
    print(f"Vector store saved to {VECTORSTORE_DIR}/")


if __name__ == "__main__":
    main()
