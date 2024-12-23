import streamlit as st
import requests
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document

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
st.sidebar.markdown("[View on GitHub](https://github.com/example-repo)")

if not user_api_key:
    st.warning("Please enter your OpenAI API Key to continue.")
    st.stop()

# LangChain Setup
embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
docs = []

# Load and process documentation
st.write("Loading documentation...")
for product, urls in documentation_urls.items():
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                # Ensure documents have metadata for source tracking
                doc.metadata["source"] = url
                docs.append(doc)
        except Exception as e:
            st.warning(f"Failed to load {url}: {e}")

# Create a vector store
st.write("Building vector store...")
vector_store = FAISS.from_documents(docs, embeddings)

# Create a Retrieval-based QA Chain
retriever = vector_store.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0, openai_api_key=user_api_key),
    retriever=retriever,
    return_source_documents=True,
)

# Streamlit Main Interface
st.title("Cloudflare Documentation Assistant")
st.write("Ask me anything about the following Cloudflare products:")
st.write(", ".join(docs_urls.keys()))

question = st.text_input("Enter your question:")
if question:
    with st.spinner("Searching documentation..."):
        try:
            result = qa_chain(question)
            st.success("Here's the answer:")
            st.write(result["result"])

            # Show sources
            st.write("Sources:")
            for doc in result["source_documents"]:
                st.write(doc.metadata["source"])
        except Exception as e:
            st.error(f"An error occurred: {e}")
