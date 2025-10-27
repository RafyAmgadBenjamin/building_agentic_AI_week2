"""
Part 1 - Web Search implementation using LangGraph with Tracing.

This implementation focuses on:
- Setting up web search using Tavily
- Processing search results
- Formatting responses with citations
- Adding Opik Tracing for observability
"""

import os
from typing import Dict, List, Optional, TypedDict
from unittest import result
from perplexia_ai.core.chat_interface import ChatInterface
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


class SearchResponse(BaseModel):
    """Model for search response with citations."""

    answer: str = Field(
        ..., description="The generated answer based on search results."
    )
    citations: List[str] = Field(
        ..., description="List of citations from search results."
    )


class SearchState(TypedDict):
    """State representation for web search workflow."""

    query: str
    search_results: List[Dict[str, str]]
    response: SearchResponse


class WebSearchChat(ChatInterface):
    """Week 2 Part 1 implementation for web search using LangGraph + Tracing."""

    def __init__(self):

        model_kwargs = {"model": "gpt-4o-mini"}
        # Get environment variables at runtime
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_api_base = os.getenv("OPENAI_API_BASE")

        if openai_api_key:
            model_kwargs["api_key"] = openai_api_key
        if openai_api_base:
            model_kwargs["base_url"] = openai_api_base

        self.llm = init_chat_model(**model_kwargs).with_structured_output(
            SearchResponse
        )
        self.search_tool = tool = TavilySearch(
            max_results=5,
            include_answer=False,  # True provides an answer from Tavily's side. Use False in assignments.
            include_raw_content=True,
            include_images=False,
            search_depth="basic",  # advanced/basic are the two options
        )
        graph_structure = StateGraph(SearchState)
        graph_structure.add_node("search", self.__search)
        graph_structure.add_node("summarize_llm", self.__summarize_llm)

        graph_structure.add_edge(START, "search")
        graph_structure.add_edge("search", "summarize_llm")
        graph_structure.add_edge("summarize_llm", END)

        self.graph = graph_structure.compile()
        self.tracer = None

    def initialize(self) -> None:
        """Initialize components for web search and tracing.

        Students should:
        - Initialize the LLM
        - Set up Tavily search tool
        - Create a LangGraph workflow for web search
        - Add Opik tracing for observability
        """
        pass

    def process_message(
        self, message: str, chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Process a message using web search and record trace.

        Args:
            message: The user's input message
            chat_history: Previous conversation history

        Returns:
            str: The assistant's response based on web search results
        """
        result = self.graph.invoke({"query": message})
        response = result["response"].dict()
        return {
            "text": f'{response["answer"]}\nSources: {", ".join(response["citations"])}'
        }

    def __search(self, state: SearchState):
        """Perform web search using the query in the state."""
        search_response = self.search_tool.invoke({"query": state["query"]})
        
        # Extract the actual results from the Tavily response
        if isinstance(search_response, dict) and "results" in search_response:
            state["search_results"] = search_response["results"]
        else:
            state["search_results"] = search_response
            
        print("Actual Search Results:", state["search_results"])
        return state

    def __summarize_llm(self, state: SearchState):
        """Generate a summarized response using LLM based on search results."""

        summarizing_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    """You are an AI assistant that analyzes web search results and provides comprehensive answers.

CRITICAL INSTRUCTIONS:
- You MUST use the search results provided below to answer the user's question
- Extract specific facts, developments, and information from the search results
- If search results contain relevant information, you MUST include it in your answer
- Only say "no information available" if the search results are truly empty or completely irrelevant
- Always cite the sources using the URLs from the search results

Your response must be in JSON format with 'answer' and 'citations' fields."""
                ),
                HumanMessagePromptTemplate.from_template(
                    """Here are the search results for the query:

{search_results}

User Query: {user_query}

Please analyze these search results and provide a comprehensive answer about {user_query}. Include specific details and developments mentioned in the search results."""
                ),
            ]
        )

        formatted_results = "\n\n".join(
            (
                f"Result {i+1}:\nTitle: {item.get('title', 'No title')}\nContent: {item.get('content', str(item))}\nURL: {item.get('url', 'No URL')}"
                if isinstance(item, dict)
                else f"Result {i+1}: {str(item)}"
            )
            for i, item in enumerate(state["search_results"])
        )

        formatted_prompt = summarizing_prompt.format(
            search_results=formatted_results,
            user_query=state["query"],
        )
        
        # Debug: Print what's being sent to LLM
        print("=== FORMATTED PROMPT FOR LLM ===")
        print(formatted_prompt)
        print("=== END PROMPT ===")
        
        response = self.llm.invoke(formatted_prompt)
        state["response"] = response
        return state
