import os
import time
import streamlit as st
from langchain_openai import ChatOpenAI
from background import Black

Black.dark_theme()

# Streamlit interface
st.title("Your Helpful Learning Assistant! �")

# Using environment variable for API key (if you set it)
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize LLM with streaming enabled (현재는 생성만 하고 사용하지 않음)
llm = ChatOpenAI(
    temperature=0.4,
    model="gpt-4o-mini",
    api_key=openai_api_key,
    streaming=True,
)

Welcome_Message = """
Kia ora! Warm Welcome to you, I am your friendly learning assistant.

Please feel free to choose Assistants from the left sidebar to help you with your queries.
"""


def stream_data(Welcome_Message):
    for word in Welcome_Message.split():
        yield word + " "
        time.sleep(0.05)


st.write("### Welcome ")
st.write_stream(stream_data(Welcome_Message))

# # ✅ Welcome Message가 한 번만 실행되도록 세션 상태 저장
# if "Welcome_Message" not in st.session_state:
#     st.session_state.Welcome_Message = False  # 처음 실행될 때만 False


# def stream_data(Welcome_Message):
#     for word in Welcome_Message.split():
#         yield word + " "
#         time.sleep(0.05)

#     # if not st.session_state.Welcome_Message:


# st.write("### Welcome ")
# st.write_stream(stream_data(Welcome_Message))


#     st.session_state.Welcome_Message = True  # 이후에는 실행되지 않음
