import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI
import json
import re

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
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

# Sidebar configuration
with st.sidebar:
    st.title("Configuration")
    st.session_state.openai_api_key = st.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        value=st.session_state.openai_api_key,
    )
    difficulty = st.select_slider(
        "Select Difficulty", options=["Easy", "Medium", "Hard"], value="Medium"
    )
    choice = st.selectbox("Choose Input Method:", ["File", "Wikipedia Article"])
    st.markdown(
        "[View on GitHub](https://github.com/LeConsulat2/gpt-2025/blob/master/pages/QuizGPT.py)"
    )


# Functions
def extract_json(content):
    """Extract JSON from a string using regex."""
    match = re.search(r"{.*}", content, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("JSON not found in response")


def generate_quiz(context, difficulty):
    """Generate quiz questions using LLM."""
    system_prompt = f"""
    Create a {difficulty.lower()}-level quiz in JSON format with 10 questions about the following content:
    {context}
    The quiz must follow this structure:
    {{
        "questions": [
            {{
                "question": "Sample question?",
                "answers": [
                    {{"answer": "Wrong", "correct": false}},
                    {{"answer": "Correct", "correct": true}}
                ]
            }}
        ]
    }}
    """
    try:
        llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-4",
            openai_api_key=st.session_state.openai_api_key,
        )
        response = llm.invoke(system_prompt)
        return extract_json(response.content)
    except Exception as e:
        st.error(f"Error generating quiz: {str(e)}")
        return {
            "questions": [
                {
                    "question": "Failed to generate quiz. Try again?",
                    "answers": [
                        {"answer": "Yes", "correct": True},
                        {"answer": "No", "correct": False},
                    ],
                }
            ]
        }


def handle_quiz_submission(answers):
    """Handle quiz submission and validate answers."""
    correct_answers = all(
        any(answer["correct"] and answer["selected"] for answer in q["answers"])
        for q in answers
    )
    st.session_state.all_correct = correct_answers
    if correct_answers:
        st.balloons()


def retrieve_wikipedia_content(query):
    """Retrieve content from Wikipedia using WikipediaRetriever."""
    retriever = WikipediaRetriever()
    try:
        content = retriever.retrieve(query)
        return content.page_content if content else "No content found."
    except Exception as e:
        st.error(f"Error retrieving Wikipedia content: {str(e)}")
        return "Error retrieving content."


# Main content
if st.session_state.openai_api_key:
    if choice == "Wikipedia Article":
        article_query = st.text_input("Enter Wikipedia Article Topic:")
        if article_query and st.button("Fetch Wikipedia Content"):
            st.session_state.current_doc = retrieve_wikipedia_content(article_query)
            if st.session_state.current_doc:
                st.success("Content retrieved successfully!")

    elif choice == "File":
        uploaded_file = st.file_uploader(
            "Upload a text file", type=["txt", "pdf", "docx"]
        )
        if uploaded_file:
            if uploaded_file.type == "text/plain":
                st.session_state.current_doc = uploaded_file.read().decode("utf-8")
            elif uploaded_file.type == "application/pdf":
                import PyPDF2

                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                st.session_state.current_doc = "\n".join(
                    page.extract_text() for page in pdf_reader.pages
                )
            elif (
                uploaded_file.type
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ):
                import docx

                doc = docx.Document(uploaded_file)
                st.session_state.current_doc = "\n".join(
                    paragraph.text for paragraph in doc.paragraphs
                )
            st.success("File uploaded and processed!")

    if st.session_state.current_doc and st.button("Generate Quiz"):
        st.session_state.quiz_state = generate_quiz(
            st.session_state.current_doc, difficulty
        )
        st.session_state.quiz_key += 1

    if st.session_state.quiz_state:
        quiz_data = st.session_state.quiz_state
        for idx, question in enumerate(quiz_data["questions"]):
            st.write(f"**Q{idx+1}: {question['question']}**")
            # Generate a unique key for each radio button
            selected_answer = st.radio(
                f"Options for Q{idx+1}:",
                options=[a["answer"] for a in question["answers"]],
                key=f"q{idx}_ans_{st.session_state.quiz_key}",
            )
            # Update selected status for each answer
            for ans in question["answers"]:
                ans["selected"] = ans["answer"] == selected_answer
