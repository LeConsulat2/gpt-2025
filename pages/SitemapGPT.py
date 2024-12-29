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

    # Create a FAISS vector store
    vector_store = FAISS.from_texts(
        texts=[doc.page_content for doc in docs if hasattr(doc, "page_content")],
        embedding=embeddings,
        metadatas=[doc.metadata for doc in docs if hasattr(doc, "metadata")],
    )

    retriever = vector_store.as_retriever()

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given a chat history and the latest user question, formulate a standalone question "
                "which can be understood without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=retriever, prompt=contextualize_q_prompt
    )

    # ---- 핵심: stuff 체인 생성 시 {context}가 들어가도록 prompt 수정 ----
    chain = create_stuff_documents_chain(
        llm=llm,
        prompt=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an assistant for question-answering tasks. "
                    "Use the following context to answer the question. "
                    "If you don't know the answer, just say that you don't know. "
                    "Use three sentences maximum and keep the answer concise.",
                ),
                # 아래처럼 {context} 변수를 꼭 사용해야 KeyError가 안 납니다.
                ("human", "{context}\n\nQuestion: {question}"),
            ]
        ),
    )

    return {
        "retriever": history_aware_retriever,
        "combine_docs_chain": chain,
    }


# Initialize chain
chain_dict = initialize_chain()

# Main interface
st.title("Cloudflare Documentation Assistant")
st.write("Ask me anything about the following Cloudflare products:")
st.write(", ".join(docs_urls.keys()))

if "messages" not in st.session_state:
    st.session_state.messages = []

question = st.text_input("Enter your question:")

if question:
    with st.spinner("Fetching the response..."):
        # 1) Retrieve relevant docs based on chat history + new question
        docs = chain_dict["retriever"].invoke({"chat_history": [], "input": question})

        valid_docs = []
        if isinstance(docs, list):
            for doc in docs:
                if isinstance(doc, str):
                    continue
                if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
                    valid_docs.append(doc)

        if valid_docs:
            # 2) Stuff chain 호출 시 문서는 input_documents -> {context}, 질문은 {question}에 매핑
            answer = chain_dict["combine_docs_chain"].invoke(
                {
                    "input_documents": valid_docs,
                    "question": question,
                }
            )

            st.success("Here's the answer:")
            st.write(answer)

            st.write("Sources:")
            sources = set(doc.metadata["source"] for doc in valid_docs)
            for source in sources:
                st.markdown(f"- [{source}]({source})")

            st.session_state.messages.append(("human", question))
            st.session_state.messages.append(("assistant", answer))
        else:
            st.error("No valid documents found or invalid document structure.")
