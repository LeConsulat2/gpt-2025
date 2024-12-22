import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from PyPDF2 import PdfReader
import tempfile
import json
import os

st.set_page_config(page_title="QuizGPT", page_icon="❓")

# Sidebar for configuration
with st.sidebar:
    st.title("Configuration")
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""

    if not st.session_state.openai_api_key:
        st.session_state.openai_api_key = st.text_input(
            "Enter your OPENAI API KEY:", type="password"
        )
        if st.session_state.openai_api_key:
            st.success("API Key saved. You can now create quizzes.")

    choice = st.selectbox("Choose Input Method:", ["File", "Wikipedia Article"])

if not st.session_state.openai_api_key:
    st.title("QuizGPT")
    st.warning("Please enter your OPENAI API KEY in the sidebar to proceed.")
else:
    llm = ChatOpenAI(
        temperature=0.5, model="gpt-4", openai_api_key=st.session_state.openai_api_key
    )

    def generate_questions(context):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                You are a teacher creating a quiz based on the provided text. Generate 10 questions, each with four options. One option should be correct, marked with (o). Example:
                Question: What is the color of the ocean?
                Answers: Red|Yellow|Green|Blue(o)

                Context: {context}
                """,
                )
            ]
        )
        chain = prompt | llm
        return chain.invoke({"context": context})  # Use `invoke` instead of `run`

    def parse_questions(questions_text):
        class JsonOutputParser(BaseOutputParser):
            def parse(self, text):
                text = text.replace("```", "").replace("json", "")
                return json.loads(text)

        output_parser = JsonOutputParser()
        return output_parser.parse(questions_text)

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
                loader = UnstructuredFileLoader(temp_file.name)
                splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                texts = loader.load_and_split(text_splitter=splitter)
        elif file_type == "txt":
            texts = file.read().decode("utf-8").splitlines()
        return "\n".join(texts)

    st.title("QuizGPT")

    docs = None
    if choice == "File":
        uploaded_file = st.file_uploader(
            "Upload a file (docx, pdf, txt):", type=["docx", "pdf", "txt"]
        )
        if uploaded_file:
            docs = process_file(uploaded_file)
    elif choice == "Wikipedia Article":
        topic = st.text_input("Search Wikipedia:")
        if topic:
            retriever = WikipediaRetriever(top_k_results=1)
            results = retriever.get_relevant_documents(topic)
            docs = results[0].page_content if results else ""

    if docs:
        questions_text = generate_questions(
            docs
        )  # Generate questions using updated `invoke`
        parsed_questions = parse_questions(questions_text)

        with st.form("quiz_form"):
            st.write("### Quiz Questions")
            for question in parsed_questions["questions"]:
                st.write(question["question"])
                options = [answer["answer"] for answer in question["answers"]]
                selected_option = st.radio(
                    "Select an option:", options, key=question["question"]
                )

            submit = st.form_submit_button("Submit")
            if submit:
                for question in parsed_questions["questions"]:
                    selected_option = st.session_state.get(question["question"])
                    correct_option = next(
                        answer["answer"]
                        for answer in question["answers"]
                        if answer["correct"]
                    )
                    if selected_option == correct_option:
                        st.success(f"Correct: {question['question']}")
                    else:
                        st.error(
                            f"Wrong: {question['question']} (Correct: {correct_option})"
                        )
