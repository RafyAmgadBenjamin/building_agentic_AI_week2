"""Part 1 - Tool-Using Agent implementation.

This implementation focuses on:
- Converting tools from Assignment 1 to use with LangGraph
- Using the ReAct pattern for autonomous tool selection
- Comparing manual workflow vs agent approaches
"""

from typing import Dict, List, Optional
from perplexia_ai.core.chat_interface import ChatInterface


class ToolUsingAgentChat(ChatInterface):
    """Week 3 Part 1 implementation focusing on tool-using agents."""
    
    def __init__(self):
        self.llm = None
        self.tools = []
        self.graph = None
    
    def initialize(self) -> None:
        """Initialize components for the tool-using agent.
        
        Students should:
        - Initialize the chat model
        - Define tools for calculator, DateTime, and weather
        - Create the ReAct agent using LangGraph
        """
        pass
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using the tool-using agent.
        
        Args:
            message: The user's input message
            chat_history: Previous conversation history
            
        Returns:
            str: The assistant's response
        """
        return "Not implemented yet. Please implement Week 3 Part 1: Tool-Using Agent."

