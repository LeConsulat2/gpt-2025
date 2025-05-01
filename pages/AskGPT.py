import os
from dotenv import load_dotenv
import logging
import streamlit as st
from background import Black
from streamlit_extras.add_vertical_space import add_vertical_space
from typing import List, Dict

from langchain.chat_models import ChatOpenAI

# 환경 변수 로드
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit 페이지 설정
st.set_page_config(page_title="GPT 4.1 AI 챗봇", page_icon="🤖", layout="wide")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# 다크 테마 적용
try:
    Black.dark_theme()
except Exception as e:
    logger.warning(f"다크 테마 적용 실패: {e}")

# LLM 초기화
if not openai_api_key:
    st.error("🚨 OPENAI_API_KEY가 설정되지 않았습니다. .env 파일에 키를 추가해주세요.")
    raise ValueError("API 키 없음")

# (Pydantic v2 대응) 모델 재빌드
try:
    ChatOpenAI.model_rebuild()
except AttributeError:
    pass

llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.3,
    api_key=openai_api_key,
    streaming=True,
)


class GPTChatAssistant:
    def __init__(self):
        self.chat = llm
        self.init_session_state()

    def init_session_state(self):
        default_system_prompt = {
            "role": "system",
            "content": (
                "당신은 유용하고 친절한 AI 어시스턴트입니다. "
                "사용자가 어떤 질문을 하더라도 명확하고 정확하게 답변하세요."
            ),
        }
        if "conversation" not in st.session_state:
            st.session_state.conversation = [default_system_prompt]
        st.session_state.chat_history = []

    def stream_response(self, conversation: List[Dict]):
        try:
            with st.spinner("답변 생성 중..."):
                response = self.chat.stream(conversation)
                full_response = ""
                for chunk in response:
                    if chunk.content:
                        full_response += chunk.content
                        yield full_response
        except Exception as e:
            logger.error(f"답변 생성 오류: {e}")
            st.error(f"답변 생성 중 오류가 발생했습니다: {e}")
            yield "죄송합니다. 요청을 처리하는 중 문제가 발생했습니다."

    def run(self):
        st.title("🤖 GPT 4.1 AI 챗봇")
        st.markdown("*무엇이든 물어보세요. OpenAI GPT-4.1 기반 AI 챗봇입니다.*")

        with st.sidebar:
            st.header("채팅 설정")
            if st.button("🔄 대화 초기화"):
                self.init_session_state()
                st.rerun()
            add_vertical_space(2)
            st.markdown("**최근 질문 기록**")
            for msg in st.session_state.chat_history[-5:]:
                st.markdown(f"- {msg}")

        for message in st.session_state.conversation[1:]:
            role = "user" if message["role"] == "user" else "assistant"
            st.chat_message(role).write(message["content"])

        if prompt := st.chat_input("무엇이든 질문해보세요..."):
            st.session_state.conversation.append({"role": "user", "content": prompt})
            st.session_state.chat_history.append(prompt)
            st.chat_message("user").write(prompt)

            response_placeholder = st.chat_message("assistant").empty()
            full_response = ""
            for response_chunk in self.stream_response(st.session_state.conversation):
                full_response = response_chunk
                response_placeholder.markdown(full_response)

            st.session_state.conversation.append(
                {"role": "assistant", "content": full_response}
            )


def main():
    try:
        assistant = GPTChatAssistant()
        assistant.run()
    except Exception as e:
        st.error(f"치명적 오류 발생: {e}")
        logger.critical(f"애플리케이션 실행 실패: {e}")


if __name__ == "__main__":
    main()
