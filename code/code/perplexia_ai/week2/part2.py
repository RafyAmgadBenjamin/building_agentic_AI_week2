"""Part 2 - Document RAG implementation using LangGraph.

This implementation focuses on:
- Setting up document loading and processing
- Creating vector embeddings and storage
- Implementing retrieval-augmented generation
- Formatting responses with citations from OPM documents
"""

import os
from typing import Dict, List, Optional, TypedDict
from perplexia_ai.core.chat_interface import ChatInterface
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
import asyncio
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END



class RAGState(TypedDict):
    query: str
    documents: list[str]
    generated_answer: str


class DocumentRAGChat(ChatInterface):
    """Week 2 Part 2 implementation for document RAG."""

    def __init__(self):
        model_kwargs = {"model": "gpt-4o-mini"}
        # Get environment variables at runtime
        openrouter_api_key = os.getenv("OPENAI_API_KEY")  # This is actually your OpenRouter key
        openai_api_base = os.getenv("OPENAI_API_BASE")
        actual_openai_key = os.getenv("ACTUAL_OPENAI_API_KEY")  # Real OpenAI key for embeddings

        if openrouter_api_key:
            model_kwargs["api_key"] = openrouter_api_key
        if openai_api_base:
            model_kwargs["base_url"] = openai_api_base

        self.llm = init_chat_model(**model_kwargs)

        # Use real OpenAI API directly for embeddings (OpenRouter doesn't support embeddings)
        print(f"🔍 Debug - actual_openai_key exists: {actual_openai_key is not None}")
        print(f"🔍 Debug - actual_openai_key length: {len(actual_openai_key) if actual_openai_key else 0}")
        
        if not actual_openai_key:
            raise ValueError("ACTUAL_OPENAI_API_KEY environment variable is required for embeddings!")
        
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=actual_openai_key,
            # Explicitly ensure we're using OpenAI's official endpoint
            base_url="https://api.openai.com/v1"  
        )
        
        persist_dir = os.path.join(os.path.dirname(__file__), "chroma_db")

        # Create vector store AFTER embeddings are properly initialized
        self.vector_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
            collection_name="opm_documents_db",
        )
        base_dir = os.path.join(os.path.dirname(__file__), "RAG_dataset")
        self.document_paths = [
            os.path.join(base_dir, "2019-annual-performance-report.pdf"),
            os.path.join(base_dir, "2020-annual-performance-report.pdf"),
            os.path.join(base_dir, "2021-annual-performance-report.pdf"),
            os.path.join(base_dir, "2022-annual-performance-report.pdf"),
        ]

        builder = StateGraph(RAGState)
        builder.add_node("load_docs", self.__load_docs)
        builder.add_node("retrieve_docs", self.__retrieve_relevant_docs)
        builder.add_node("generate_answer", self.__generate_answer)

        builder.add_edge(START, "load_docs")
        builder.add_edge("load_docs", "retrieve_docs")
        builder.add_edge("retrieve_docs", "generate_answer")
        builder.add_edge("generate_answer", END)


        self.graph = builder.compile()
        print("✅ LangGraph initialized successfully.")


    def initialize(self) -> None:
        """Initialize components for document RAG.

        Students should:
        - Initialize the LLM
        - Set up document loading and processing (e.g., OPM annual reports)
        - Create vector embeddings
        - Build retrieval system
        - Create LangGraph for RAG workflow
        """

    def process_message(
        self, message: str, chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Process a message using document RAG.

        Should retrieve relevant information from documents and generate responses.

        Args:
            message: The user's input message
            chat_history: Previous conversation history

        Returns:
            str: The assistant's response based on document knowledge
        """
        result = self.graph.invoke({"query": message})
        return result["generated_answer"]

    def __load_docs(self, state: RAGState):
        """Load the documents for RAG if it is not loaded yet."""
        try:
            # Use Chroma's get method to check if documents exist
            existing_data = self.vector_store.get(limit=1)
            if existing_data['ids']:  # If there are any document IDs
                print("✅ Vector store already initialized with documents. Skipping re-embedding.")
                return state
            else:
                print("📄 Loading and embedding OPM documents...")
                self.__load_embed_documents()
        except Exception as e:
            print(f"⚠️ Error checking vector store, re-initializing: {e}")
            print("📄 Loading and embedding OPM documents...")
            self.__load_embed_documents()
        return state

    def __load_embed_documents(self):
        """Load the documents for RAG."""
        all_pages = []

        async def load_all():
            for pdf_file in self.document_paths:
                if os.path.exists(pdf_file):
                    loader = PyPDFLoader(pdf_file)
                    async for page in loader.alazy_load():
                        all_pages.append(page)
                else:
                    print(f"Warning: PDF file not found: {pdf_file}")
            print(f"Loaded {len(all_pages)} pages from documents.")

        try:
            asyncio.run(load_all())
            print(f"Total pages loaded: {len(all_pages)}")

            if all_pages:
                # Add the documents directly to the vector store
                self.vector_store.add_documents(all_pages)
                print("Documents embedded into vector store.")
            else:
                print("No documents loaded. Please check your PDF files.")
        except Exception as e:
            print(f"Error loading/embedding documents: {e}")
            raise

    def __retrieve_relevant_docs(self, state: RAGState, k: int = 4) -> List:
        """Retrieve relevant documents from vector store based on query."""
        if not self.vector_store:
            raise ValueError("Vector store is not initialized.")
        state["documents"] = self.vector_store.similarity_search(state["query"], k=k)
        return state

    def __generate_answer(self, state: RAGState):
        """Generate answer using LLM and retrieved documents."""
        if not state["documents"]:
            return "No relevant documents found."
        prompt = PromptTemplate.from_template(
            "Answer the question based on the context provided. \n"
            "Question: {question}\n"
            "Context: {context}\n"
            "Include the source of the information in your answer, If you don't know the answer, just say you don't know." 
        )
        context = "\n".join([doc.page_content for doc in state["documents"]])
        prompt = prompt.format(question=state["query"], context=context)
        state["generated_answer"] = self.llm.invoke(prompt).content
        return state
