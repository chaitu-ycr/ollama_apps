import os
import logging
from typing import Iterator, List
import ollama
from dotenv import load_dotenv


class OllamaAPIHandler:
    """
    Handles interactions with the Ollama API, including fetching models and streaming chat responses.
    """

    def __init__(self):
        self.config = self._load_configuration()

    @staticmethod
    def _load_configuration() -> dict:
        """
        Load API configuration from environment variables.
        """
        load_dotenv()
        return {
            "api_base_url": os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434"),
            "default_model": os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.2"),
        }

    def get_available_models(self) -> List[str]:
        """
        Fetch the list of available models from the Ollama API.
        """
        os.environ["OLLAMA_API_BASE_URL"] = self.config["api_base_url"]
        try:
            response = ollama.list()
            return [model.model for model in response.models if model.model]
        except AttributeError as error:
            logging.error("The 'ollama' module does not support fetching models: %s", error)
            return []
        except Exception as error:
            logging.exception("Failed to fetch models: %s", error)
            return []

    def stream_chat_responses(self, model_name: str, user_message: str) -> Iterator[str]:
        """
        Stream chat responses for a given user message and model.
        """
        user_message = user_message.strip()
        if not user_message:
            raise ValueError("User message cannot be empty.")

        api_base_url = self.config.get("api_base_url")
        if not api_base_url:
            logging.error("API base URL is not configured.")
            raise RuntimeError("API base URL is not configured.")

        os.environ["OLLAMA_API_BASE_URL"] = api_base_url

        try:
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": user_message}],
                stream=True
            )
            for chunk in response:
                content = chunk.get("message", {}).get("content")
                if content:
                    yield content
        except ollama.OllamaError as api_error:
            logging.error("Ollama API error: %s", api_error)
            raise RuntimeError("Failed to communicate with Ollama API.") from api_error
        except Exception as error:
            logging.exception("Unexpected error during API communication: %s", error)
            raise RuntimeError("An unexpected error occurred.") from error