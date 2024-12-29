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
    """Load documentation URLs for specified products from the sitemap."""
    response = requests.get(sitemap_url)
    response.raise_for_status()

    # Parse the XML sitemap
    sitemap = response.text
    product_urls = {product: [] for product in products}

    for product, base_url in products.items():
        # Filter URLs containing the product's base URL
        product_urls[product] = [
            line.split("<loc>")[1].split("</loc>")[0]
            for line in sitemap.splitlines()
            if base_url in line
        ]

    return product_urls


# Load the documentation URLs
documentation_urls = load_documentation_urls(sitemap_url, docs_urls)

# Streamlit Sidebar
st.sidebar.title("SiteGPT for Cloudflare Docs")
user_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
st.sidebar.markdown(
    "[View on GitHub](https://github.com/LeConsulat2/gpt-2025/blob/master/pages/SitemapGPT.py)"
)

if not user_api_key:
    st.warning("Please enter your OpenAI API Key to continue.")
    st.stop()

# LangChain Setup
embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
docs = []

# Load and process documentation
for product, urls in documentation_urls.items():
    for url in urls:
        loader = WebBaseLoader(url)
        data = loader.load()

        # Split the loaded documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(data)

        for split in all_splits:
            split.metadata["source"] = url
            docs.append(split)

# Extract text and metadata for embedding
texts = [doc.page_content for doc in docs]
metadata = [doc.metadata for doc in docs]

# Create a vector store
vector_store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadata)

# Contextualize question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
retriever = vector_store.as_retriever()
history_aware_retriever = create_history_aware_retriever(
    ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        openai_api_key=user_api_key,
    ),
    retriever,
    contextualize_q_prompt,
)

# Answer question
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    openai_api_key=user_api_key,
)
retrieval_chain = create_retrieval_chain(
    llm=llm,
    retriever=history_aware_retriever,
    combine_documents_chain=create_stuff_documents_chain(
        llm=llm,
        prompt=ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("input_documents"),
                ("human", "{input}"),
            ]
        ),
    ),
)

# Streamlit Main Interface
st.title("Cloudflare Documentation Assistant")
st.write("Ask me anything about the following Cloudflare products:")
st.write(", ".join(docs_urls.keys()))

# User Query
question = st.text_input("Enter your question:")
placeholder = st.empty()
if question:
    with st.spinner("Fetching the response..."):
        result = retrieval_chain({"input": question})
        placeholder.success("Here's the answer:")
        placeholder.write(result["output"])

        # Show sources with distinct URLs
        st.write("Sources:")
        sources = set(doc.metadata["source"] for doc in result["source_documents"])
        for source in sources:
            st.markdown(f"- [{source}]({source})")
