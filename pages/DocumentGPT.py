import streamlit as st
import nltk
from langchain_openai import ChatOpenAI
from langchain_unstructured import UnstructuredLoader
from unstructured.cleaners.core import clean_extra_whitespace
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
import tempfile
import io
from background import Black

# Apply Black theme
Black.dark_theme()

# App Title and Description
st.title("DocumentGPT")
st.markdown("Upload your documents and ask questions about them.")
st.divider()

# Initialize session state for messages
if "doc_messages" not in st.session_state:
    st.session_state.doc_messages = []

# File uploader for documents
uploaded_files = st.file_uploader(
    "Upload your documents", type=["pdf", "docx", "txt"], accept_multiple_files=True
)

# Initialize the OpenAI Chat Model
chat = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    streaming=True,
)

# Process uploaded files
if uploaded_files:
    with st.spinner("Processing documents..."):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )

        all_texts = []

        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            # Use UnstructuredLoader for all file types with lazy loading
            loader = UnstructuredLoader(
                file_path=temp_file_path, post_processors=[clean_extra_whitespace]
            )
            # Use lazy_load for memory efficiency
            texts = loader.lazy_load()

            # Split text lazily
            for text in texts:
                splits = text_splitter.split_text(text.page_content)
                for split in splits:
                    all_texts.append(split)

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vectorstore = InMemoryVectorStore.from_documents(
            [{"page_content": text} for text in all_texts], embeddings
        )

        # Save to session state
        st.session_state.vectorstore = vectorstore
        st.success(f"Successfully processed {len(uploaded_files)} documents")

# Chat interface remains the same as before
if "vectorstore" in st.session_state:
    for message in st.session_state.doc_messages:
        if isinstance(message, HumanMessage):
            st.chat_message("user").write(message.content)
        else:
            st.chat_message("assistant").write(message.content)

    query = st.chat_input("Ask a question about your documents")

    if query:
        user_message = HumanMessage(content=query)
        st.session_state.doc_messages.append(user_message)
        st.chat_message("user").write(query)

        # Search similar content
        similar_docs = st.session_state.vectorstore.similarity_search(query, k=3)
        context = "\n".join(doc.page_content for doc in similar_docs)

        prompt = f"""Based on the following context, please answer the question. 
        If you cannot find the answer in the context, say so.
        
        Context: {context}
        
        Question: {query}"""

        response_placeholder = st.chat_message("assistant").empty()
        full_response = ""

        messages = [HumanMessage(content=prompt)]
        for chunk in chat.stream(messages):
            if chunk.content:
                full_response += chunk.content
                response_placeholder.markdown(full_response)

        st.session_state.doc_messages.append(AIMessage(content=full_response))
