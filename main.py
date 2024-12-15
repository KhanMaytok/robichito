import os
import sys
import argparse
from typing import List

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


class OllamaRAGChat:
    def __init__(self, documents_path: str, model_name: str = "llama3.1:8b", embedding_model: str = "llama3.1:8b"):
        """
        Initialize RAG Chat with document indexing and Ollama model

        :param documents_path: Path to directory containing documents
        :param model_name: Ollama model name for text generation
        :param embedding_model: Ollama model for embeddings
        """
        self.documents_path = documents_path
        self.persist_dir = os.path.join(documents_path, ".index")

        # Setup Ollama LLM and Embedding
        self.llm = Ollama(model=model_name, request_timeout=120.0)
        self.embedding = OllamaEmbedding(model_name=embedding_model)

        # Initialize index
        self.index = self._load_or_create_index()

        # Create query engine
        self.query_engine = self.index.as_query_engine(
            llm=self.llm,
            similarity_top_k=3  # Retrieve top 3 most relevant documents
        )

    def _load_or_create_index(self):
        """
        Load existing index or create a new one if not exists

        :return: VectorStoreIndex
        """
        if os.path.exists(self.persist_dir):
            try:
                # Reload from persistent storage
                storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
                return load_index_from_storage(storage_context)
            except:
                print("Could not load existing index. Creating a new one.")

        # Create new index
        print(f"Indexing documents from {self.documents_path}")
        documents = SimpleDirectoryReader(self.documents_path).load_data()
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=self.embedding
        )

        # Persist index
        index.storage_context.persist(persist_dir=self.persist_dir)
        return index

    def chat(self):
        """
        Interactive chat interface with context retrieval
        """
        print("\n🤖 Michifuzo 🤖")
        print("Type 'exit' or 'quit' to end the conversation.\n")

        conversation_history = []
        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break

                # Add context from vectorstore
                response = self.query_engine.query(user_input)

                print("\nAssistant:", response.response)
                print("\n📚 Context Sources:")
                for node in response.source_nodes:
                    print(f"- Relevance Score: {node.score:.2f}")
                    print(f"  Source: {node.node.metadata.get('file_name', 'Unknown')}")

                print("\n" + "-" * 50 + "\n")

            except KeyboardInterrupt:
                print("\nChat interrupted. Type 'exit' to quit.")
            except Exception as e:
                print(f"An error occurred: {e}")


def main():
    parser = argparse.ArgumentParser(description="Ollama RAG Chat with Document Context")
    documents_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/llamita/blog_posts"
    print(documents_path)

    parser.add_argument(
        "--model",
        default="llama3.1:8b",
        help="Ollama model name for text generation"
    )

    args = parser.parse_args()

    chat_system = OllamaRAGChat(
        documents_path=documents_path,
        model_name=args.model
    )

    chat_system.chat()


if __name__ == "__main__":
    main()
