import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

# 문서 URL 정의
docs_urls = {
    "AI Gateway": "https://developers.cloudflare.com/ai-gateway/",
    "Cloudflare Vectorize": "https://developers.cloudflare.com/vectorize/",
    "Workers AI": "https://developers.cloudflare.com/workers-ai/",
}

# Streamlit 인터페이스 설정
st.title("Cloudflare Documentation Assistant")
st.sidebar.title("SiteGPT for Cloudflare Docs")
user_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
st.sidebar.markdown(
    "[View on GitHub](https://github.com/LeConsulat2/gpt-2025/blob/master/pages/SitemapGPT.py)"
)

if not user_api_key:
    st.warning("Please enter your OpenAI API Key to continue.")
    st.stop()

# 전역 설정
llm = ChatOpenAI(temperature=0.3, openai_api_key=user_api_key)
embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)


@st.cache_resource
def initialize_vector_store():
    docs = []
    for url in docs_urls.values():
        loader = WebBaseLoader(url)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(data)
        docs.extend(splits)

    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    return FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)


def get_response(question, vector_store):
    docs = vector_store.similarity_search(question)

    prompt = PromptTemplate(
        template="""Answer the following question based on the provided context. 
        If you don't know the answer, just say you don't know.
        
        Context: {context}
        Question: {question}
        
        Answer:""",
        input_variables=["context", "question"],
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    context = "\n".join([doc.page_content for doc in docs])
    response = chain.run(context=context, question=question)

    return response, docs


# 메인 인터페이스
st.write("Ask me anything about the following Cloudflare products:")
st.write(", ".join(docs_urls.keys()))

vector_store = initialize_vector_store()

# 사용자 입력 및 응답
question = st.text_input("Enter your question:")
if question:
    with st.spinner("Searching for an answer..."):
        response, docs = get_response(question, vector_store)

        st.success("Answer:")
        st.write(response)

        st.write("\nSources:")
        sources = set(doc.metadata["source"] for doc in docs)
        for source in sources:
            st.markdown(f"- [{source}]({source})")
