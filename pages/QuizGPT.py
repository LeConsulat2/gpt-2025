import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain.llms import OpenAI
from docx import Document
from PyPDF2 import PdfReader
import tempfile


def generate_question(difficulty):
    """Generate a question based on difficulty."""
    if difficulty == "Easy":
        return {
            "question": "What is 2 + 2?",
            "options": ["3", "4", "5", "6"],
            "answer": "4",
        }
    elif difficulty == "Hard":
        return {
            "question": "What is the derivative of x^2?",
            "options": ["2x", "x^2", "x", "1"],
            "answer": "2x",
        }


def process_file(file):
    """Process uploaded file and extract text."""
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
    return "\n".join(texts)


# Sidebar for navigation and authentication
with st.sidebar:
    st.title("Settings")
    openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    github_link = "https://github.com/your-repo-link"
    st.markdown(f"[View on GitHub]({github_link})")
    navigation = st.radio("Navigate to:", ["App", "QuizGPT"])

if navigation == "App":
    st.title("Welcome to the Main App")
    if not openai_api_key:
        st.warning(
            "Please enter your OpenAI API Key in the sidebar to proceed to QuizGPT."
        )
    else:
        st.success("API Key authenticated! Switch to QuizGPT from the sidebar.")

if navigation == "QuizGPT" and openai_api_key:
    llm = OpenAI(api_key=openai_api_key, temperature=0.5)

    st.title("QuizGPT")

    choice = st.selectbox(
        "Choose Input Method:", ["Wikipedia Search", "Upload Document"]
    )

    if choice == "Wikipedia Search":
        topic = st.text_input("Enter a Wikipedia topic:")
        if topic:
            retriever = WikipediaRetriever(top_k_results=1)
            documents = retriever.get_relevant_documents(topic)
            st.session_state.input_text = (
                documents[0].page_content if documents else "No content found."
            )
            st.write(st.session_state.input_text)

    elif choice == "Upload Document":
        uploaded_file = st.file_uploader(
            "Upload a file (docx, pdf, txt):", type=["docx", "pdf", "txt"]
        )
        if uploaded_file:
            st.session_state.input_text = process_file(uploaded_file)
            st.write(st.session_state.input_text)

    # Show difficulty selection and generate quiz only if input text exists
    if "input_text" in st.session_state and st.session_state.input_text:
        difficulty = st.selectbox("Select Difficulty:", ["Easy", "Hard"])

        # Generate a question
        if "quiz_data" not in st.session_state:
            st.session_state.quiz_data = generate_question(difficulty)
            st.session_state.correct = False

        quiz_data = st.session_state.quiz_data

        # Display question and options
        st.subheader(quiz_data["question"])

        selected_option = st.radio("Choose your answer:", quiz_data["options"])

        if st.button("Submit Answer"):
            if selected_option == quiz_data["answer"]:
                st.success("Correct Answer!")
                st.session_state.correct = True
                st.balloons()
            else:
                st.error("Wrong Answer. Try again.")
                st.session_state.correct = False

        if st.session_state.correct and st.button("Retake Test"):
            st.session_state.quiz_data = generate_question(difficulty)
            st.session_state.correct = False
