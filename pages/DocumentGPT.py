import streamlit as st
import tempfile
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from background import Black
from docx import Document
from pypdf import PdfReader

Black.dark_theme()

# App UI
st.title("DocumentGPT")
st.markdown("Upload your documents and ask questions about them.")
st.divider()

# Initialize session state
if "doc_messages" not in st.session_state:
    st.session_state.doc_messages = []

# Initialize AI models
chat = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    streaming=True,
)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


# File processing
def process_file(file):
    file_type = file.name.split(".")[-1]
    texts = []

    if file_type == "pdf":
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            texts.append(page.extract_text())
    elif file_type == "docx":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
            temp_file.write(file.read())
            temp_file.flush()
            doc = Document(temp_file.name)
            for para in doc.paragraphs:
                texts.append(para.text)
    elif file_type == "txt":
        texts = file.read().decode("utf-8").splitlines()
    else:
        raise ValueError("Unsupported file type.")

    return texts


# File upload
uploaded_files = st.file_uploader(
    "Upload your documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

# Process documents
if uploaded_files:
    with st.spinner("Processing documents..."):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=50,
        )

        all_texts = []
        for file in uploaded_files:
            texts = process_file(file)
            for text in texts:
                all_texts.extend(text_splitter.split_text(text))

        st.session_state.vectorstore = FAISS.from_texts(all_texts, embeddings)
        st.success(f"Processed {len(uploaded_files)} documents")

# Chat interface
if "vectorstore" in st.session_state:
    # Show chat history
    for message in st.session_state.doc_messages:
        with st.chat_message(
            "user" if isinstance(message, HumanMessage) else "assistant"
        ):
            st.write(message.content)

    # Handle new questions
    query = st.chat_input("Ask a question about your documents")
    if query:
        st.session_state.doc_messages.append(HumanMessage(content=query))

        # Get relevant context
        similar_docs = st.session_state.vectorstore.similarity_search(query, k=3)
        context = "\n".join(doc.page_content for doc in similar_docs)
        prompt = f"""Based on the following context, please answer the question. 
        If you cannot find the answer in the context, say so.
        
        Context: {context}
        
        Question: {query}"""

        # Stream response
        response_placeholder = st.chat_message("assistant").empty()
        full_response = ""
        for chunk in chat.stream([HumanMessage(content=prompt)]):
            if chunk.content:
                full_response += chunk.content
                response_placeholder.markdown(full_response)

        st.session_state.doc_messages.append(AIMessage(content=full_response))
