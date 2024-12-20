from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import streamlit as st
import os
import time
from docx2txt import process
from PyPDF2 import PdfReader
from background import Black

Black.dark_theme()

# Streamlit interface
st.title("FilesGPT")

Welcome_Message = """
Hello!! You got in here, which means you do have an OPENAI API KEY. 
Streamlit is easy to use and I wish all other apps were as easy to use as this one.
"""


def stream_data():
    for word in Welcome_Message.split():
        yield word + " "
        time.sleep(0.05)


# Sidebar configuration
st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("Enter your OPENAI API KEY", type="password")

# Check for API key
if "api_key_entered" not in st.session_state:
    st.session_state.api_key_entered = False

if api_key:
    if not st.session_state.api_key_entered:
        st.write("### Welcome Message")
        st.write_stream(stream_data)
        st.session_state.api_key_entered = True

    # Initialize LLM with streaming
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-4",
        openai_api_key=api_key,
        streaming=True,  # Enable streaming
    )
else:
    st.error("Please enter your OPENAI API KEY in the sidebar")
    st.stop()

with st.sidebar:
    st.markdown(
        "[Github Code](https://github.com/LeConsulat2/gpt-2025/blob/master/App.py)"
    )


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
        # Use docx2txt for DOCX files
        text = process(temp_file_path)
        documents.append(
            Document(page_content=text, metadata={"source": temp_file_path})
        )
    elif file_extension == "pdf":
        # Use PyPDF2 for PDF files
        reader = PdfReader(temp_file_path)
        text = "\n".join([page.extract_text() for page in reader.pages])
        documents.append(
            Document(page_content=text, metadata={"source": temp_file_path})
        )
    else:
        # Use UnstructuredFileLoader for TXT and other file types
        from langchain.document_loaders import UnstructuredFileLoader

        loader = UnstructuredFileLoader(temp_file_path)
        loaded_documents = loader.load()
        documents.extend(
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in loaded_documents
        )

    # Initialize embeddings and split documents
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
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

        # Stream response from OpenAI
        with st.chat_message("assistant"):
            response_container = st.empty()
            response_text = ""
            for chunk in llm.stream({"context": prompt}):  # Streaming enabled in llm
                response_text += chunk["text"]
                response_container.markdown(response_text)  # Update dynamically

            # Save final response
            st.session_state.messages.append(
                {"role": "assistant", "content": response_text}
            )

    # Cleanup
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)
