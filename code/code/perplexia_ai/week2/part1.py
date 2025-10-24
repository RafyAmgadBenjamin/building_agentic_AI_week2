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
from perplexia_ai.core.chat_interface import ChatInterface
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
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

        self.llm = init_chat_model(
            **model_kwargs, reasoning_effects="minimal"
        ).with_structured_output(SearchResponse)
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
        return "Not implemented yet. Please implement Week 2 Part 1: Web Search with LangGraph."

    def __search(self, state: SearchState):
        """Perform web search using the query in the state."""
        state["search_results"] = self.search_tool.invoke({"query": state["query"]})
        return state

    def __summarize_llm(self, state: SearchState):
        """Generate a summarized response using LLM based on search results."""

        summarizing_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    """You are an AI assistant that provides concise answers based on web search results.
                Given the following search results, generate a clear and concise answer to the user's query.
                You don’t need to cite a document for each sentence. The format should be an 'answer' and a list of 'sources' at the end.
                The summarized search results should appear in the 'answer', while all citations should be listed in 'sources'.

                Examples:

                Search Results:
                1. "SpaceX successfully launched Starship today at 10:00 AM..."
                2. "The launch marks a milestone in commercial space travel..."

                User Query: When did SpaceX launch Starship today?

                Answer: SpaceX successfully launched Starship today at 10:00 AM, marking a significant milestone in commercial space travel.
                Sources: ["https://example.com/article1", "https://example.com/article2"]

                Search Results:
                1. "The new climate policy focuses on reducing carbon emissions..."
                2. "It introduces incentives for renewable energy adoption..."

                User Query: What are the key points of the new climate policy?

                Answer: The new climate policy aims to reduce carbon emissions and provides incentives for adopting renewable energy.
                Sources: ["https://example.com/climate1", "https://example.com/climate2"]
                """
                ),
                HumanMessagePromptTemplate.from_template(
                    """Now, based on the search results below, generate the answer in the same format:

                Search Results:
                {search_results}

                User Query:
                {user_query}

                Answer:"""
                ),
            ]
        )

        formatted_prompt = summarizing_prompt.format(
            search_results="\n".join(state.search_results),
            user_query=state.query
        )
        response = self.llm.invoke(formatted_prompt)
        state.response = response
        return state
