# Phase 1 LLM Notes

This sample document is here so the chatbot has something to answer questions
about before you add your own notes or PDF.

The project uses retrieval augmented generation, often called RAG. RAG combines
a search step with a generation step. First, the app searches for relevant text
chunks from a document. Then, it sends those chunks to a language model along
with the user's question.

The main parts of this project are:

- A document loader, which reads a PDF, Markdown file, or text file.
- A text splitter, which breaks the document into overlapping chunks.
- An embedding model, which turns chunks into vectors.
- A vector database, which stores and searches those vectors.
- A chat model, which answers questions using the retrieved context.

For a first project, the chatbot should answer only from the document. If the
answer is not present in the document, it should say that it does not know.

