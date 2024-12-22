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
    You must respond with valid JSON only.
    Create a {difficulty.lower()}-level quiz with exactly this JSON structure:
    {{
        "questions": [
            {{
                "question": "What is an example question?",
                "answers": [
                    {{"answer": "Wrong answer 1", "correct": false}},
                    {{"answer": "Correct answer", "correct": true}},
                    {{"answer": "Wrong answer 2", "correct": false}},
                    {{"answer": "Wrong answer 3", "correct": false}}
                ]
            }}
        ]
    }}
    
    Rules for {difficulty.lower()} difficulty:
    - Easy: Basic recall and simple comprehension questions
    - Medium: Understanding and application questions
    - Hard: Analysis and evaluation questions with more complex options
    
    Create 10 questions about this content: {context}
    Important: Return ONLY the JSON object with no additional text or formatting.
    """

    try:
        llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-4",
            openai_api_key=st.session_state.openai_api_key,
        )

        response = llm.invoke(system_prompt)

        # Try to parse the JSON response
        try:
            quiz_data = json.loads(response.content)
            # Validate the quiz structure
            if not isinstance(quiz_data, dict) or "questions" not in quiz_data:
                raise ValueError("Invalid quiz structure")
            return quiz_data
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON from the response
            content = response.content
            # Find the first { and last } to extract just the JSON part
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end != 0:
                try:
                    quiz_data = json.loads(content[start:end])
                    if not isinstance(quiz_data, dict) or "questions" not in quiz_data:
                        raise ValueError("Invalid quiz structure")
                    return quiz_data
                except:
                    pass

            # If all parsing attempts fail, return a default quiz
            return {
                "questions": [
                    {
                        "question": "Error generating quiz. Would you like to try again?",
                        "answers": [
                            {"answer": "Yes, regenerate the quiz", "correct": True},
                            {"answer": "No, keep these questions", "correct": False},
                            {
                                "answer": "Try with different difficulty",
                                "correct": False,
                            },
                            {"answer": "Try with different content", "correct": False},
                        ],
                    }
                ]
            }
    except Exception as e:
        st.error(f"Error generating quiz: {str(e)}")
        return {
            "questions": [
                {
                    "question": "Error generating quiz. Would you like to try again?",
                    "answers": [
                        {"answer": "Yes, regenerate the quiz", "correct": True},
                        {"answer": "No, keep these questions", "correct": False},
                        {"answer": "Try with different difficulty", "correct": False},
                        {"answer": "Try with different content", "correct": False},
                    ],
                }
            ]
        }


def regenerate_quiz():
    """Regenerate the quiz with proper error handling"""
    try:
        st.session_state.quiz_key += 1
        if st.session_state.current_doc:
            new_quiz = generate_quiz(st.session_state.current_doc, difficulty)
            if new_quiz and "questions" in new_quiz:
                st.session_state.quiz_state = new_quiz
            else:
                st.error("Failed to generate new questions. Please try again.")
    except Exception as e:
        st.error(f"Error regenerating quiz: {str(e)}")
