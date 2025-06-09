"""
Module to start the Ollama Streamlit web application.
"""

import os
import sys
import socket
import subprocess
import logging
from pathlib import Path
from typing import Optional
import argparse


class RunWebApp:
    """Handles configuration, environment setup, CLI argument parsing, and launching the Streamlit web app."""

    @staticmethod
    def configure_logging(log_level: str = "INFO") -> None:
        """
        Configure logging for the application.

        Args:
            log_level (str): Logging level. Defaults to "INFO".
        """
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format="%(levelname)s: %(message)s"
        )
        logging.info("Logging configured with level: %s", log_level)

    @staticmethod
    def set_environment_variables(parallel: int = 4, max_queue: int = 64) -> None:
        """
        Set required environment variables for the application.

        Args:
            parallel (int): Number of parallel processes. Defaults to 4.
            max_queue (int): Maximum queue size. Defaults to 64.
        """
        os.environ["OLLAMA_NUM_PARALLEL"] = str(parallel)
        os.environ["OLLAMA_MAX_QUEUE"] = str(max_queue)
        logging.info("Environment variables set: OLLAMA_NUM_PARALLEL=%d, OLLAMA_MAX_QUEUE=%d", parallel, max_queue)

    @staticmethod
    def parse_args() -> argparse.Namespace:
        """
        Parse CLI arguments for server address and port.

        Returns:
            argparse.Namespace: Parsed arguments containing server address and port.
        """
        parser = argparse.ArgumentParser(description="Start the Ollama Streamlit web application.")
        parser.add_argument("--server-address", type=str, default=None, help="Bind address (default: system FQDN)")
        parser.add_argument("--server-port", type=int, default=8501, help="Port (default: 8501)")
        args = parser.parse_args()
        logging.info("Parsed arguments: server_address=%s, server_port=%d", args.server_address, args.server_port)
        return args

    @staticmethod
    def launch(address: Optional[str] = None, port: int = 8501) -> bool:
        """
        Start the Streamlit web app. Returns True if successful.

        Args:
            address (Optional[str]): The server address to bind to. Defaults to the system FQDN.
            port (int): The port to bind the server to. Defaults to 8501.

        Returns:
            bool: True if the web app starts successfully, False otherwise.
        """
        app_file = Path(__file__).parent / "web_app.py"
        if not app_file.is_file():
            logging.error("web_app.py not found at %s", app_file)
            return False

        bind_address = address or socket.getfqdn()
        cmd = [
            sys.executable, "-m", "streamlit", "run", str(app_file),
            "--server.address", bind_address,
            "--server.port", str(port)
        ]
        logging.debug("Running command: %s", ' '.join(cmd))

        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logging.info("Streamlit app started at http://%s:%d", bind_address, port)
            return True
        except FileNotFoundError:
            logging.error("Streamlit executable not found. Ensure Streamlit is installed.")
            return False
        except Exception as exc:
            logging.exception("Failed to start Streamlit app: %s", exc)
            return False

    @classmethod
    def run(cls) -> None:
        """
        Main entry point for starting the Ollama Streamlit web application.
        """
        try:
            # Configure logging
            log_level = os.environ.get("OLLAMA_APPS_LOGLEVEL", "INFO")
            cls.configure_logging(log_level)

            # Set environment variables
            cls.set_environment_variables(parallel=8, max_queue=1024)

            # Parse CLI arguments
            args = cls.parse_args()

            # Launch the web application
            if not cls.launch(args.server_address, args.server_port):
                logging.error("Failed to launch the web application.")
                sys.exit(1)
        except KeyboardInterrupt:
            logging.info("Application interrupted by user.")
            sys.exit(0)
        except Exception as exc:
            logging.exception("An unexpected error occurred: %s", exc)
            sys.exit(1)


if __name__ == '__main__':
    RunWebApp.run()