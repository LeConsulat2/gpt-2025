import os
import time
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from docx2txt import process
from PyPDF2 import PdfReader
from background import Black

Black.dark_theme()

# Streamlit interface
st.title("DocumentGPT")

# Using environment variable for API key (if you set it)
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize LLM with streaming enabled
llm = ChatOpenAI(
    temperature=0.4,
    model="gpt-4o-mini",
    api_key=openai_api_key,
    streaming=True,
)


Welcome_Message = """
Kia ora! Warm Welcome to you, I am your friendly learning assistant.
Please feel free to ask me anything about any queries you may have!
You can also choose other types of Assistants from the left sidebar to help you with your queries.
"""

# ✅ Welcome Message가 한 번만 실행되도록 세션 상태 저장
if "Welcome_Message" not in st.session_state:
    st.session_state.Welcome_Message = False  # 처음 실행될 때만 False


def stream_data(message):
    for word in message.split():
        yield word + " "
        time.sleep(0.05)


if not st.session_state.Welcome_Message:
    st.write("### Welcome ")
    st.write_stream(stream_data(Welcome_Message))
    st.session_state.Welcome_Message = True  # 이후에는 실행되지 않음


def create_chain(retriever, llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Use the following context to answer the question: {context}"),
            ("human", "{question}"),
        ]
    )

    def retrieve_and_format(input_dict):
        question = input_dict["question"]
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        return {"context": context, "question": question}

    chain = retrieve_and_format | prompt | llm | StrOutputParser()
    return chain


# File upload
uploaded_file = st.file_uploader("Upload your document", type=["txt", "pdf", "docx"])

if uploaded_file:
    # Process uploaded file
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Determine file type and process
    file_extension = uploaded_file.name.split(".")[-1].lower()
    documents = []

    if file_extension == "docx":
        text = process(temp_file_path)
        documents.append(
            Document(page_content=text, metadata={"source": temp_file_path})
        )
    elif file_extension == "pdf":
        reader = PdfReader(temp_file_path)
        text = "\n".join([page.extract_text() for page in reader.pages])
        documents.append(
            Document(page_content=text, metadata={"source": temp_file_path})
        )
    else:
        from langchain.document_loaders import UnstructuredFileLoader

        loader = UnstructuredFileLoader(temp_file_path)
        loaded_documents = loader.load()
        documents.extend(
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in loaded_documents
        )

    # Initialize embeddings and split documents
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)

    # Create vector store and retriever
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

    # Create the chain
    chain = create_chain(retriever, llm)

    # Display chat history and handle user input
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about your document"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Streaming logic
        response_placeholder = st.chat_message("assistant").empty()
        full_response = ""

        for chunk in chain.stream({"question": prompt}):
            full_response = full_response + chunk
            response_placeholder.markdown(full_response)  # Update Progressively

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

    # Cleanup
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)
