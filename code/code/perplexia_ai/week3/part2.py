"""Part 2 - Agentic RAG implementation.

This implementation focuses on:
- Building an agent that can dynamically control its search strategy
- Combining document retrieval with web search
- Making autonomous decisions about information gathering
"""

from typing import Dict, List, Optional
from perplexia_ai.core.chat_interface import ChatInterface


class AgenticRAGChat(ChatInterface):
    """Week 3 Part 2 implementation focusing on Agentic RAG."""
    
    def __init__(self):
        self.llm = None
        self.vector_store = None
        self.search_tool = None
        self.graph = None
    
    def initialize(self) -> None:
        """Initialize components for Agentic RAG.
        
        Students should:
        - Initialize the chat model
        - Set up document vector store with OPM documents
        - Create tools for document retrieval and web search
        - Build an agent that can autonomously decide which tools to use
        """
        pass
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using the Agentic RAG system.
        
        Args:
            message: The user's input message
            chat_history: Previous conversation history
            
        Returns:
            str: The assistant's response
        """
        return "Not implemented yet. Please implement Week 3 Part 2: Agentic RAG."

