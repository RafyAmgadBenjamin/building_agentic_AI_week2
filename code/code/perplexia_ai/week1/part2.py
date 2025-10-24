"""Part 2 - Basic Tools implementation using LangGraph.

This implementation focuses on:
- Detect when calculations are needed
- Use calculator for mathematical operations
- Format calculation results clearly
"""

from typing import Dict, List, Optional

from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.tools.calculator import Calculator


class BasicToolsChat(ChatInterface):
    """Week 1 Part 2 implementation adding calculator functionality using LangGraph."""
    
    def __init__(self):
        """Initialize components for basic tools using LangGraph.
        
        Students should:
        - Initialize the chat model
        - Build a graph with classifier, response, and tool nodes
        - Add calculator and datetime tool nodes
        - Set up conditional edges for routing
        - Compile the graph
        """
        # TODO: Students implement initialization
        self.llm = None
        self.graph = None
        self.calculator = Calculator()
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message with calculator support using LangGraph.
        
        Students should:
        - Initialize state with the question
        - Invoke the graph
        - Extract and return the response
        
        Args:
            message: The user's input message
            chat_history: Not used in Part 2
            
        Returns:
            str: The assistant's response
        """
        # TODO: Students implement calculator integration using LangGraph
        return "Not implemented yet"

