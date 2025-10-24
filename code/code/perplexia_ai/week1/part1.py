"""Part 1 - Query Understanding implementation using LangGraph.

This implementation focuses on:
- Classify different types of questions
- Format responses based on query type using conditional edges
- Present information professionally
"""

from typing import Dict, List, Optional

from perplexia_ai.core.chat_interface import ChatInterface


class QueryUnderstandingChat(ChatInterface):
    """Week 1 Part 1 implementation focusing on query understanding using LangGraph."""
    
    def __init__(self):
        """Initialize components for query understanding using LangGraph.
        
        Students should:
        - Initialize the chat model
        - Build a graph with classifier and response nodes
        - Set up conditional edges for routing based on query category
        - Compile the graph
        """
        # TODO: Students implement initialization
        self.llm = None
        self.graph = None
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using query understanding with LangGraph.
        
        Students should:
        - Initialize state with the question
        - Invoke the graph
        - Extract and return the response
        
        Args:
            message: The user's input message
            chat_history: Not used in Part 1
            
        Returns:
            str: The assistant's response
        """
        # TODO: Students implement query understanding using LangGraph
        return "Not implemented yet"

