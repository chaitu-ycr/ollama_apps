import logging
import os
from typing import List

import streamlit as st
from dotenv import load_dotenv

from ollama_apps.ollama_api_handler import OllamaAPIHandler
from ollama_apps.document_processor import DocumentProcessor


class OllamaChatApplication:
    """
    Manages the Streamlit UI for the Ollama chat application.
    """

    def __init__(self):
        self.api_handler = OllamaAPIHandler()
        self.processor = DocumentProcessor()
        self._setup_logging()

    @staticmethod
    def _setup_logging(level: int = logging.INFO) -> None:
        """
        Configure logging for the application.
        """
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(message)s"
        )
        logging.info("Logging initialized.")

    @staticmethod
    def _initialize_chat_history() -> List[dict]:
        """
        Retrieve or initialize chat history data in session state.
        """
        return st.session_state.setdefault("chat_history", [])

    def _render_live_chat_tab(self, selected_model: str):
        """
        Render the Live Chat tab with user input, send button, and live response.
        """
        user_message = st.text_area("Enter your message:", placeholder="Type your message here...")
        if st.button("Send"):
            user_message = user_message.strip()
            if not user_message:
                st.warning("Please enter a message.")
                return

            accumulated_response = ""
            try:
                st.write("### Live Response:")
                response_placeholder = st.empty()
                for chunk in self.api_handler.stream_chat_responses(selected_model, user_message):
                    accumulated_response += chunk
                    response_placeholder.write(accumulated_response)
                self._initialize_chat_history().append({
                    "model_name": selected_model,
                    "user_message": user_message,
                    "ollama_response": accumulated_response
                })
            except ValueError as error:
                logging.error("Validation error during live chat: %s", error)
                st.error("Invalid input. Please check your message.")
            except RuntimeError as error:
                logging.error("Runtime error during live chat: %s", error)
                st.error("An error occurred. Please check the logs.")
            except Exception as error:
                logging.exception("Unexpected error during live chat: %s", error)
                st.error("An unexpected error occurred. Please check the logs.")

    def _render_chat_history_tab(self):
        """
        Render the Chat History tab with stored user messages and responses in dictionary format.
        Display the latest query at the top.
        """
        st.write("### Chat History:")
        chat_history = self._initialize_chat_history()
        if chat_history:
            for entry in reversed(chat_history):
                st.json(entry)
        else:
            st.write("No chat history available.")

    def _render_training_tab(self):
        """
        Render the Training tab with file upload and URL input for scraping content.
        """
        st.write("### Train Local Documents")

        uploaded_files = st.file_uploader("Upload files for training:", type=["txt", "pdf", "docx", "xlsx", "csv", "pptx"], accept_multiple_files=True)
        url = st.text_input("Enter a URL for scraping content:")

        if st.button("Train"):
            if uploaded_files or url:
                file_paths = [file.name for file in uploaded_files]
                self.processor.file_paths = file_paths
                self.processor.url = url

                try:
                    self.processor.load_documents()
                    self.processor.split_documents()
                    self.processor.create_embeddings()
                    self.processor.setup_qa_chain(model=self.api_handler.config["default_model"])
                    st.success("Training completed successfully!")
                except ValueError as error:
                    logging.error("Validation error during training: %s", error)
                    st.error("Invalid input provided. Please check your files or URL.")
                except RuntimeError as error:
                    logging.error("Runtime error during training: %s", error)
                    st.error("An error occurred during training. Please check the logs.")
                except Exception as error:
                    logging.exception("Unexpected error during training: %s", error)
                    st.error("An unexpected error occurred during training. Please check the logs.")
            else:
                st.warning("Please upload files or provide a URL.")

    def run(self):
        """
        Run the Streamlit chat application.
        """
        try:
            st.set_page_config(layout="wide")

            if "session_id" not in st.session_state:
                st.session_state.session_id = os.urandom(8).hex()

            with st.sidebar:
                st.markdown("<h2>Settings</h2>", unsafe_allow_html=True)
                models = self.api_handler.get_available_models()
                selected_model = st.selectbox(
                    "Select Model",
                    models,
                    index=models.index(self.api_handler.config["default_model"]) if self.api_handler.config["default_model"] in models else 0
                )

            tabs = st.tabs(["Live Chat", "Chat History", "Train Documents"])

            with tabs[0]:
                self._render_live_chat_tab(selected_model)

            with tabs[1]:
                self._render_chat_history_tab()

            with tabs[2]:
                self._render_training_tab()
        except Exception as error:
            logging.exception("Unexpected error in application run: %s", error)
            st.error("An unexpected error occurred. Please check the logs.")


def main() -> None:
    """
    Main entry point for the Ollama Streamlit chat application.
    """
    try:
        app = OllamaChatApplication()
        app.run()
    except KeyboardInterrupt:
        logging.info("Application interrupted by user.")
    except Exception as error:
        logging.exception("An unexpected error occurred: %s", str(error))


if __name__ == "__main__":
    main()
