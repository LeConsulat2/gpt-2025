import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import background.Black as Black
from typing import List, Dict
import asyncio
from streamlit_extras.add_vertical_space import add_vertical_space
import logging

# Streamlit page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="AUT Intelligent Assistant", page_icon="🎓", layout="wide"
)


# Advanced logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


# Enhanced theme application with error handling
Black.dark_theme()
# Robust environment variable loading
load_dotenv(override=True)


class AUTChatAssistant:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7):
        """
        Initialize the AUT Chat Assistant with configurable parameters.

        :param model: OpenAI model to use
        :param temperature: Creativity/randomness of responses
        """
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            st.error("🚨 OpenAI API Key not found. Please check your environment.")
            raise ValueError("Missing API Key")

        self.chat = ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=True,
            api_key=self.openai_api_key,
        )

        self.init_session_state()

    def init_session_state(self):
        """Initialize and reset session states."""
        default_system_prompt = {
            "role": "system",
            "content": "You are an expert assistant specializing in AUT University Learning Management System. Provide precise, comprehensive answers.",
        }

        if "conversation" not in st.session_state:
            st.session_state.conversation = [default_system_prompt]

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

    def stream_response(self, conversation: List[Dict]):
        """
        Stream response with advanced error handling and timeout.

        :param conversation: Conversation history
        :return: Generated response
        """
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
        """Main application runner."""
        st.title("🎓 AUT Intelligent Chat Assistant")
        st.markdown("*Powered by Advanced AI Technologies*")

        # Sidebar with enhanced features
        with st.sidebar:
            st.header("Chat Controls")
            if st.button("🔄 Reset Conversation"):
                self.init_session_state()
                st.experimental_rerun()

            add_vertical_space(2)
            st.markdown("**Chat History**")
            for msg in st.session_state.chat_history[-5:]:
                st.markdown(f"- {msg}")

        # Main chat interface
        for message in st.session_state.conversation[1:]:
            role = "user" if message["role"] == "user" else "assistant"
            st.chat_message(role).write(message["content"])

        # User input handling
        if prompt := st.chat_input("Ask anything about AUT University"):
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
