"""Part 3 - Conversation Memory implementation using LangGraph.

This implementation focuses on:
- Maintain context across messages
- Handle follow-up questions
- Use conversation history in responses
"""

from typing import Dict, List, Optional

from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.tools.calculator import Calculator


class MemoryChat(ChatInterface):
    """Week 1 Part 3 implementation adding conversation memory using LangGraph."""
    
    def __init__(self):
        """Initialize components for memory-enabled chat using LangGraph.
        
        Students should:
        - Initialize the chat model
        - Build a graph with classifier, response, and tool nodes
        - Include history in prompts for context-aware responses
        - Set up conditional edges for routing
        - Compile the graph
        """
        # TODO: Students implement initialization with memory
        self.llm = None
        self.graph = None
        self.calculator = Calculator()
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message with memory and tools using LangGraph.
        
        Students should:
        - Format chat_history as a string
        - Initialize state with question and history
        - Invoke the graph
        - Extract and return the response
        
        Args:
            message: The user's input message
            chat_history: List of previous chat messages with 'role' and 'content' keys
            
        Returns:
            str: The assistant's response
        """
        # TODO: Students implement memory integration using LangGraph
        return "Not implemented yet"

