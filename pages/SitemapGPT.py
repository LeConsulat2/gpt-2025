import streamlit as st
import requests
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
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


# Load and process documentation
@st.cache_resource
def initialize_retrieval_chain():
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

    # Create vector store
    vector_store = FAISS.from_texts(
        texts=[doc.page_content for doc in docs],
        embedding=embeddings,
        metadatas=[doc.metadata for doc in docs],
    )

    # Setup retriever with chat history
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

    # Create the final chain
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant for question-answering tasks. Use the following context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    return create_retrieval_chain(
        llm=llm,
        retriever=history_aware_retriever,
        combine_documents_chain=create_stuff_documents_chain(llm=llm, prompt=qa_prompt),
    )


# Initialize chain
chain = initialize_retrieval_chain()

# Main interface
st.title("Cloudflare Documentation Assistant")
st.write("Ask me anything about the following Cloudflare products:")
st.write(", ".join(docs_urls.keys()))

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# User input
question = st.text_input("Enter your question:")
if question:
    with st.spinner("Fetching the response..."):
        result = chain.invoke(
            {"input": question, "chat_history": st.session_state.messages}
        )

        st.success("Here's the answer:")
        st.write(result["answer"])

        st.write("Sources:")
        sources = set(doc.metadata["source"] for doc in result["documents"])
        for source in sources:
            st.markdown(f"- [{source}]({source})")

        st.session_state.messages.append(("human", question))
        st.session_state.messages.append(("assistant", result["answer"]))
