from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import streamlit as st


VECTORSTORE_DIR = "vectorstore"

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a careful document Q&A assistant.

Answer using only the provided context.
If the answer is not in the context, say you do not know.
Keep answers concise and cite the source filename when useful.

Context:
{context}""",
        ),
        ("human", "{question}"),
    ]
)


def format_docs(docs):
    formatted = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown source")
        page = doc.metadata.get("page")
        location = f"{source}, page {page + 1}" if page is not None else source
        formatted.append(f"[{index}] Source: {location}\n{doc.page_content}")

    return "\n\n".join(formatted)


@st.cache_resource
def load_retriever():
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings,
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def answer_question(question):
    retriever = load_retriever()
    docs = retriever.invoke(question)
    context = format_docs(docs)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = model.invoke(PROMPT.format_messages(context=context, question=question))

    return response.content, docs


def main():
    load_dotenv()

    st.set_page_config(page_title="Document Chatbot")
    st.title("Document Chatbot")

    st.caption("Ask questions about the documents indexed from the data folder.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your document")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document..."):
            answer, docs = answer_question(question)

        st.markdown(answer)

        with st.expander("Sources"):
            for index, doc in enumerate(docs, start=1):
                source = doc.metadata.get("source", "unknown source")
                page = doc.metadata.get("page")
                label = f"{source}, page {page + 1}" if page is not None else source
                st.markdown(f"**{index}. {label}**")
                st.write(doc.page_content[:800])

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
