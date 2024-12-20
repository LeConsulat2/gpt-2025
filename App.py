from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import streamlit as st
import os
import time
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

    # Initialize LLM after API key is received
    llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini", openai_api_key=api_key)
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

    try:
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        # Create retriever
        loader = UnstructuredFileLoader(temp_file_path)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever()

        # Create the chain
        chain = create_chain(retriever, llm)

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask about your document"):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get AI response
            with st.chat_message("assistant"):
                response = chain.invoke({"question": prompt})
                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

    finally:
        # Cleanup
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
