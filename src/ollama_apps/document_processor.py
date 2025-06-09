import os
import logging
from langchain.document_loaders import (
    UnstructuredPDFLoader, UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader, UnstructuredPowerPointLoader,
    TextLoader
)
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders.web_base import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA


class DocumentProcessor:
    def __init__(self, file_paths=None, url=None):
        self.file_paths = file_paths or []
        self.url = url
        self.documents = []
        self.chunks = []
        self.db = None
        self.qa = None
        self._setup_logging()

    @staticmethod
    def _setup_logging(level=logging.INFO):
        """
        Configure logging for the processor.
        """
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(message)s"
        )
        logging.info("DocumentProcessor initialized.")

    def load_documents(self):
        """
        Load documents from file paths and URL.
        """
        loaders = {
            '.pdf': UnstructuredPDFLoader,
            '.docx': UnstructuredWordDocumentLoader,
            '.xlsx': UnstructuredExcelLoader,
            '.csv': CSVLoader,
            '.pptx': UnstructuredPowerPointLoader,
            '.txt': TextLoader
        }

        if not self.file_paths and not self.url:
            logging.warning("No file paths or URL provided for document loading.")
            return

        try:
            for file_path in self.file_paths:
                ext = os.path.splitext(file_path)[1].lower()
                loader_class = loaders.get(ext)
                if loader_class:
                    self.documents.extend(loader_class(file_path).load())
                    logging.info("Loaded document: %s", file_path)
                else:
                    logging.warning("Unsupported file type: %s", file_path)

            if self.url:
                self.documents.extend(WebBaseLoader(self.url).load())
                logging.info("Loaded documents from URL: %s", self.url)

        except Exception as e:
            logging.exception("Error loading documents: %s", e)
            raise RuntimeError("Failed to load documents.") from e

    def split_documents(self):
        """
        Split documents into smaller chunks for processing.
        """
        if not self.documents:
            logging.warning("No documents to split.")
            return

        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            self.chunks = splitter.split_documents(self.documents)
            logging.info("Documents split into %d chunks.", len(self.chunks))
        except Exception as e:
            logging.exception("Error splitting documents: %s", e)
            raise RuntimeError("Failed to split documents.") from e

    def create_embeddings(self):
        """
        Create embeddings for document chunks.
        """
        if not self.chunks:
            logging.warning("No chunks available for embedding creation.")
            return

        try:
            embedding = HuggingFaceEmbeddings()
            self.db = Chroma.from_documents(self.chunks, embedding)
            logging.info("Embeddings created and stored in Chroma database.")
        except Exception as e:
            logging.exception("Error creating embeddings: %s", e)
            raise RuntimeError("Failed to create embeddings.") from e

    def setup_qa_chain(self, model="llama3"):
        """
        Set up the QA chain using the specified model.
        """
        if not self.db:
            logging.error("No database available for QA chain setup.")
            raise ValueError("Database is not initialized. Create embeddings first.")

        try:
            self.llm = Ollama(model=model)
            self.qa = RetrievalQA.from_chain_type(llm=self.llm, retriever=self.db.as_retriever())
            logging.info("QA chain set up with model: %s", model)
        except Exception as e:
            logging.exception("Error setting up QA chain: %s", e)
            raise RuntimeError("Failed to set up QA chain.") from e

    def ask_question(self, query):
        """
        Ask a question using the QA chain.
        """
        if not self.qa:
            logging.error("QA chain is not set up. Call setup_qa_chain() first.")
            raise ValueError("QA chain is not set up. Call setup_qa_chain() first.")

        try:
            response = self.qa.run(query)
            logging.info("Query executed successfully: %s", query)
            return response
        except Exception as e:
            logging.exception("Error executing query: %s", e)
            raise RuntimeError("Failed to execute query.") from e