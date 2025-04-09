import os
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import background.Black as Black

# Apply theme
Black.dark_theme()

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

ChatOpenAI.model_rebuild()  # 추가

# Initialize chat model
chat = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    streaming=True,
)


# Initialize vector store
@st.cache_resource
def init_vector_store():
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    if os.path.exists("conversation_store"):
        return FAISS.load_local("conversation_store", embeddings)
    return FAISS.from_texts(["Initial conversation"], embeddings)


vector_store = init_vector_store()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful assistant for AUT University.")
    ]

# Sidebar for conversation history
with st.sidebar:
    st.title("Conversation History")
    if len(st.session_state.messages) > 1:  # If there are conversations
        for msg in st.session_state.messages[1:]:  # Skip system message
            with st.expander(
                f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'} Message"
            ):
                st.write(msg.content)

# Main chat interface
st.title("AUT Chat Assistant")
st.write("Ask any question about AUT University below!")

# Display chat messages
for msg in st.session_state.messages[1:]:  # Skip system message
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# Chat input
prompt = st.chat_input("Type your question here...")

if prompt:
    # Add user message
    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)
    st.chat_message("user").write(prompt)

    # Get similar past conversations
    similar_conversations = vector_store.similarity_search(prompt, k=3)

    # Display AI response
    response_placeholder = st.chat_message("assistant").empty()
    full_response = ""

    # Stream the response
    for chunk in chat.stream(st.session_state.messages):
        if chunk.content:
            full_response += chunk.content
            response_placeholder.markdown(full_response)

    # Save AI response
    ai_message = AIMessage(content=full_response)
    st.session_state.messages.append(ai_message)

    # Save to vector store
    conversation_text = f"User: {prompt}\nAssistant: {full_response}"
    metadata = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    vector_store.add_texts(texts=[conversation_text], metadatas=[metadata])
    vector_store.save_local("conversation_store")

    # Update sidebar with similar conversations
    with st.sidebar:
        st.subheader("Related Conversations")
        for doc in similar_conversations:
            with st.expander(
                f"Similar Conversation from {doc.metadata.get('timestamp', 'Unknown')}"
            ):
                st.write(doc.page_content)
