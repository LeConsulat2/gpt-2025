import streamlit as st
import requests
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

# Constants for the documentation URLs
docs_urls = {
    "AI Gateway": "https://developers.cloudflare.com/ai-gateway/",
    "Cloudflare Vectorize": "https://developers.cloudflare.com/vectorize/",
    "Workers AI": "https://developers.cloudflare.com/workers-ai/",
}
sitemap_url = "https://developers.cloudflare.com/sitemap-0.xml"


def load_documentation_urls(sitemap_url, products):
    response = requests.get(sitemap_url)
    sitemap = response.text
    product_urls = {product: [] for product in products}

    for product, base_url in products.items():
        product_urls[product] = [
            line.split("<loc>")[1].split("</loc>")[0]
            for line in sitemap.splitlines()
            if base_url in line
        ]

    return product_urls


# Streamlit Interface Setup
st.sidebar.title("SiteGPT for Cloudflare Docs")
user_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
st.sidebar.markdown(
    "[View on GitHub](https://github.com/LeConsulat2/gpt-2025/blob/master/pages/SitemapGPT.py)"
)

if not user_api_key:
    st.warning("Please enter your OpenAI API Key to continue.")
    st.stop()


@st.cache_resource
def initialize_chain():
    embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
    llm = ChatOpenAI(temperature=0.3, openai_api_key=user_api_key)
    docs = []

    documentation_urls = load_documentation_urls(sitemap_url, docs_urls)

    for product, urls in documentation_urls.items():
        for url in urls:
            loader = WebBaseLoader(url)
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50
            )
            all_splits = text_splitter.split_documents(data)

            for split in all_splits:
                split.metadata["source"] = url
                docs.append(split)

    vector_store = FAISS.from_texts(
        texts=[doc.page_content for doc in docs],
        embedding=embeddings,
        metadatas=[doc.metadata for doc in docs],
    )

    retriever = vector_store.as_retriever()

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given a chat history and the latest user question, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=retriever, prompt=contextualize_q_prompt
    )

    chain = create_stuff_documents_chain(
        llm=llm,
        prompt=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an assistant for question-answering tasks. Use the following context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.",
                ),
                ("human", "{context}\nQuestion: {input}"),
            ]
        ),
    )

    return {"retriever": history_aware_retriever, "combine_docs_chain": chain}


# Initialize chain
chain_dict = initialize_chain()

# Main interface
st.title("Cloudflare Documentation Assistant")
st.write("Ask me anything about the following Cloudflare products:")
st.write(", ".join(docs_urls.keys()))

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
question = st.text_input("Enter your question:")
if question:
    with st.spinner("Fetching the response..."):
        # Stream documents with history-aware retriever
        retriever_input = {
            "chat_history": st.session_state.chat_history,
            "input": question,
        }

        # Stream documents
        doc_stream = chain_dict["retriever"].stream(retriever_input)
        docs = []

        for doc in doc_stream:
            # 디버깅용 출력: 반환된 데이터 타입과 내용을 확인
            st.write(f"Retrieved document: {doc} (Type: {type(doc)})")

            # 문자열이 반환될 경우 무시하거나 로깅
            if isinstance(doc, str):
                st.write(f"Skipped unexpected string: {doc}")
                continue

            # 올바른 객체인지 확인
            if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
                docs.append(doc)
            else:
                st.write(f"Invalid object structure: {doc}")

        # Ensure docs are valid
        if docs:
            context = "\n\n".join(
                [doc.page_content for doc in docs if hasattr(doc, "page_content")]
            )
        else:
            context = "No relevant documents found."

        # Stream the answer
        answer_stream = chain_dict["combine_docs_chain"].stream(
            {"context": context, "input": question}
        )
        st.success("Here's the answer:")
        answer_text = ""

        # Debugging: output the structure and type of each chunk
        for chunk in answer_stream:
            st.write(f"Answer chunk: {chunk} (Type: {type(chunk)})")

            # 문자열만 처리
            if isinstance(chunk, str):
                answer_text += chunk
                st.write(answer_text)  # Dynamically update the UI
            else:
                st.write(f"Unexpected non-string chunk: {chunk}")

        # Display sources
        st.write("Sources:")
        if docs:
            sources = set(
                doc.metadata["source"] for doc in docs if hasattr(doc, "metadata")
            )
            for source in sources:
                st.markdown(f"- [{source}]({source})")
        else:
            st.write("No sources available.")

        # Update chat history
        st.session_state.chat_history.append({"role": "human", "content": question})
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer_text}
        )
