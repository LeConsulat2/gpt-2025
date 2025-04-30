import os
from dotenv import load_dotenv
import logging
import streamlit as st
from background import Black
from streamlit_extras.add_vertical_space import add_vertical_space
from typing import List, Dict
from langchain.chat_models import ChatOpenAI

# Rebuild Pydantic models to avoid `not fully defined` errors
ChatOpenAI.model_rebuild()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.4,
    api_key=os.getenv("OPENAI_API_KEY"),
    streaming=True,
)

# Streamlit page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="AUT Intelligent Assistant", page_icon="🎓", layout="wide"
)

# Advanced logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Apply dark theme with error handling
try:
    Black.dark_theme()
except Exception as e:
    logger.warning(f"Failed to apply theme: {e}")

# Load environment variables (override existing)
load_dotenv(override=True)


class AUTChatAssistant:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            st.error("🚨 OpenAI API Key not found. Please check your .env file.")
            raise ValueError("Missing OPENAI_API_KEY")

        # Initialize the ChatOpenAI with streaming
        self.chat = ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=True,
            api_key=self.openai_api_key,
        )
        self.init_session_state()

    def init_session_state(self):
        default_system_prompt = {
            "role": "system",
            "content": (
                "You are an expert assistant specializing in "
                "AUT University Learning Management System. Provide precise, comprehensive answers."
            ),
        }
        if "conversation" not in st.session_state:
            st.session_state.conversation = [default_system_prompt]
        st.session_state.chat_history = []

    def stream_response(self, conversation: List[Dict]):
        try:
            with st.spinner("Generating response..."):
                response = self.chat.stream(conversation)
                full_response = ""
                for chunk in response:
                    if chunk.content:
                        full_response += chunk.content
                        yield full_response
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            st.error(f"Error generating response: {e}")
            yield "I apologize, but I encountered an error processing your request."

    def run(self):
        st.title("🎓 AUT Intelligent Chat Assistant")
        st.markdown("*Powered by Advanced AI Technologies*")

        with st.sidebar:
            st.header("Chat Controls")
            if st.button("🔄 Reset Conversation"):
                self.init_session_state()
                st.rerun()
            add_vertical_space(2)
            st.markdown("**Chat History**")
            for msg in st.session_state.chat_history[-5:]:
                st.markdown(f"- {msg}")

        for message in st.session_state.conversation[1:]:
            role = "user" if message["role"] == "user" else "assistant"
            st.chat_message(role).write(message["content"])

        if prompt := st.chat_input("Ask anything about AUT University Learning..."):
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
        assistant = AUTChatAssistant()
        assistant.run()
    except Exception as e:
        st.error(f"Critical error: {e}")
        logger.critical(f"Application failed to start: {e}")


if __name__ == "__main__":
    main()
