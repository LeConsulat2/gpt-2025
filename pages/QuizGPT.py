import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI
import tempfile
import json
import os
from PyPDF2 import PdfReader
import docx
import io

st.set_page_config(page_title="QuizGPT", page_icon="❓")

# Initialize session states
if "quiz_state" not in st.session_state:
    st.session_state.quiz_state = None
if "all_correct" not in st.session_state:
    st.session_state.all_correct = False
if "current_doc" not in st.session_state:
    st.session_state.current_doc = None
if "quiz_key" not in st.session_state:
    st.session_state.quiz_key = 0

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
    Generate completely new and different questions each time.
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
    - Hard: Analysis and evaluation questions with more complex options
    
    Generate 10 different questions based on this context: {context}
    Make sure to randomize the order of questions and answers each time.
    """

    llm = ChatOpenAI(
        temperature=0.9,  # Increased for more variety
        model="gpt-4o-mini",
        openai_api_key=st.session_state.openai_api_key,
    )

    response = llm.invoke(system_prompt)
    return json.loads(response.content)


def read_docx(file):
    doc = docx.Document(file)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return "\n".join(text)


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
        elif file_type == "docx":
            # Create a temporary file to save the uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file.flush()
                text = read_docx(tmp_file.name)
            # Clean up the temporary file
            os.unlink(tmp_file.name)
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
        return text
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None


def regenerate_quiz():
    st.session_state.quiz_key += 1
    if st.session_state.current_doc:
        st.session_state.quiz_state = generate_quiz(
            st.session_state.current_doc, difficulty
        )


def main():
    st.title("QuizGPT")

    if not st.session_state.openai_api_key:
        st.warning("Please enter your OpenAI API Key in the sidebar to proceed.")
        return

    docs = None
    if choice == "File":
        uploaded_file = st.file_uploader(
            "Upload a file (pdf, docx, txt):", type=["pdf", "docx", "txt"]
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
        # Store the current document
        st.session_state.current_doc = docs

        # Generate new quiz if needed
        if st.session_state.quiz_state is None:
            st.session_state.quiz_state = generate_quiz(docs, difficulty)

        # Add a button to generate new questions
        if st.button("Generate New Questions"):
            regenerate_quiz()

        with st.form(f"quiz_form_{st.session_state.quiz_key}"):
            st.write(f"### {difficulty} Level Quiz")
            correct_count = 0
            total_questions = len(st.session_state.quiz_state["questions"])

            for question in st.session_state.quiz_state["questions"]:
                st.write(question["question"])
                options = [answer["answer"] for answer in question["answers"]]
                selected_option = st.radio(
                    "Select an option:",
                    options,
                    key=f"{question['question']}_{st.session_state.quiz_key}",
                )

            submit = st.form_submit_button("Submit")

            if submit:
                for question in st.session_state.quiz_state["questions"]:
                    selected_option = st.session_state[
                        f"{question['question']}_{st.session_state.quiz_key}"
                    ]
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
                        regenerate_quiz()
                        st.experimental_rerun()


if __name__ == "__main__":
    main()
