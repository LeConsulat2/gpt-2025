import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain.chat_models import ChatOpenAI
import tempfile
import json
import os
from PyPDF2 import PdfReader
import io

st.set_page_config(page_title="QuizGPT", page_icon="❓")

# Initialize session states
if "quiz_state" not in st.session_state:
    st.session_state.quiz_state = None
if "all_correct" not in st.session_state:
    st.session_state.all_correct = False

# Sidebar configuration
with st.sidebar:
    st.title("Configuration")

    # API Key input
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""

    st.session_state.openai_api_key = st.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        value=st.session_state.openai_api_key,
    )

    # Difficulty selector
    difficulty = st.select_slider(
        "Select Difficulty", options=["Easy", "Medium", "Hard"], value="Medium"
    )

    # Input method selector
    choice = st.selectbox("Choose Input Method:", ["File", "Wikipedia Article"])

    # GitHub repo link
    st.markdown("[View on GitHub](https://github.com/yourusername/quizgpt)")


def generate_quiz(context, difficulty):
    """Generate quiz questions using function calling"""
    system_prompt = f"""
    You are a teacher creating a {difficulty.lower()}-level quiz.
    Generate questions as a JSON object with the following structure:
    {{
        "questions": [
            {{
                "question": "question text",
                "answers": [
                    {{"answer": "option 1", "correct": false}},
                    {{"answer": "option 2", "correct": true}},
                    {{"answer": "option 3", "correct": false}},
                    {{"answer": "option 4", "correct": false}}
                ]
            }}
        ]
    }}
    
    For {difficulty.lower()} difficulty:
    - Easy: Basic recall and simple comprehension questions
    - Medium: Understanding and application questions
    - Hard: Analysis and evaluation questions
    
    Generate 10 questions based on this context: {context}
    """

    llm = ChatOpenAI(
        temperature=0.5, model="gpt-4", openai_api_key=st.session_state.openai_api_key
    )

    response = llm.invoke(system_prompt)
    return json.loads(response.content)


def process_file(file):
    """Process uploaded files"""
    text = ""
    file_type = file.name.split(".")[-1].lower()

    try:
        if file_type == "pdf":
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        elif file_type == "txt":
            text = file.getvalue().decode("utf-8")
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
        return text
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None


def main():
    st.title("QuizGPT")

    if not st.session_state.openai_api_key:
        st.warning("Please enter your OpenAI API Key in the sidebar to proceed.")
        return

    docs = None
    if choice == "File":
        uploaded_file = st.file_uploader(
            "Upload a file (pdf, txt):", type=["pdf", "txt"]
        )
        if uploaded_file:
            docs = process_file(uploaded_file)
    else:
        topic = st.text_input("Search Wikipedia:")
        if topic:
            retriever = WikipediaRetriever(top_k_results=1)
            results = retriever.get_relevant_documents(topic)
            docs = results[0].page_content if results else ""

    if docs:
        if st.session_state.quiz_state is None:
            st.session_state.quiz_state = generate_quiz(docs, difficulty)

        with st.form("quiz_form"):
            st.write(f"### {difficulty} Level Quiz")
            correct_count = 0
            total_questions = len(st.session_state.quiz_state["questions"])

            for question in st.session_state.quiz_state["questions"]:
                st.write(question["question"])
                options = [answer["answer"] for answer in question["answers"]]
                selected_option = st.radio(
                    "Select an option:", options, key=question["question"]
                )

            submit = st.form_submit_button("Submit")

            if submit:
                for question in st.session_state.quiz_state["questions"]:
                    selected_option = st.session_state[question["question"]]
                    correct_option = next(
                        answer["answer"]
                        for answer in question["answers"]
                        if answer["correct"]
                    )
                    if selected_option == correct_option:
                        st.success(f"Correct: {question['question']}")
                        correct_count += 1
                    else:
                        st.error(
                            f"Wrong: {question['question']} (Correct: {correct_option})"
                        )

                if correct_count == total_questions:
                    st.balloons()
                    st.success("🎉 Perfect Score! Congratulations!")
                else:
                    st.warning(f"Score: {correct_count}/{total_questions}")
                    if st.button("Retake Quiz"):
                        st.session_state.quiz_state = generate_quiz(docs, difficulty)
                        st.experimental_rerun()


if __name__ == "__main__":
    main()
