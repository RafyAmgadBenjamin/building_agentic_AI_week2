"""Part 1 - Query Understanding implementation using LangGraph.

This implementation focuses on:
- Classify different types of questions
- Format responses based on query type using conditional edges
- Present information professionally
"""

from typing import Dict, List, Optional, TypedDict, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END
from perplexia_ai.core.chat_interface import ChatInterface

# Classifier prompt
CLASSIFIER_PROMPT = ChatPromptTemplate.from_template("""
Classify the given user question into one of the specified categories based on its nature.

- Factual Questions: Questions starting with phrases like "What is...?" or "Who invented...?" should be classified as 'factual'.
- Analytical Questions: Questions starting with phrases like "How does...?" or "Why do...?" should be classified as 'analytical'.
- Comparison Questions: Questions starting with phrases like "What's the difference between...?" should be classified as 'comparison'.
- Definition Requests: Questions starting with phrases like "Define..." or "Explain..." should be classified as 'definition'.

If the question does not fit into any of these categories, return 'default'.

# Steps

1. Analyze the user question.
2. Determine which category the question fits into based on its structure and keywords.
3. Return the corresponding category or 'default' if none apply.

# Output Format

- Return only the category word: 'factual', 'analytical', 'comparison', 'definition', or 'default'.
- Do not include any extra text or quotes in the output.

# Examples

- **Example 1**
* Question: What is the highest mountain in the world?  
* Response: factual

- **Example 2**  
* Question: What's the difference between OpenAI and Anthropic?  
* Response: comparison

User question: {question}
""")

# Response prompts for each category
RESPONSE_PROMPTS = {
    "factual": ChatPromptTemplate.from_template(
        """
        Answer the following question concisely with a direct fact. Avoid unnecessary details.

        User question: "{question}"
        Answer:
        """
    ),
    "analytical": ChatPromptTemplate.from_template(
        """
        Provide a detailed explanation with reasoning for the following question. Break down the response into logical steps.

        User question: "{question}"
        Explanation:
        """
    ),
    "comparison": ChatPromptTemplate.from_template(
        """
        Compare the following concepts. Present the answer in a structured format using bullet points or a table for clarity.

        User question: "{question}"
        Comparison:
        """
    ),
    "definition": ChatPromptTemplate.from_template(
        """
        Define the following term and provide relevant examples and use cases for better understanding.

        User question: "{question}"
        Definition:
        Examples:
        Use Cases:
        """
    ),
    "default": ChatPromptTemplate.from_template(
        """
        Respond your best to answer the following question but keep it very brief.

        User question: "{question}"
        Answer:
        """
    )
}

# State definition for the graph
class QueryState(TypedDict):
    """State for the query understanding graph."""
    question: str
    category: str
    response: str

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
        # Initialize the LLM
        self.llm = init_chat_model("gpt-5-mini", model_provider="openai", reasoning_effort='minimal')
        
        # Build the graph
        self.graph = None
        self._build_graph()
    
    def _build_graph(self) -> None:
        """Build the LangGraph with conditional routing."""
        # Create the graph
        workflow = StateGraph(QueryState)
        
        # Add classifier node
        workflow.add_node("classifier", self._classify_query)
        
        # Add response nodes for each category
        workflow.add_node("factual_response", self._factual_response)
        workflow.add_node("analytical_response", self._analytical_response)
        workflow.add_node("comparison_response", self._comparison_response)
        workflow.add_node("definition_response", self._definition_response)
        workflow.add_node("default_response", self._default_response)
        
        # Set entry point
        workflow.set_entry_point("classifier")
        
        # Add conditional edges from classifier to appropriate response nodes
        workflow.add_conditional_edges(
            "classifier",
            self._route_query,
            {
                "factual": "factual_response",
                "analytical": "analytical_response",
                "comparison": "comparison_response",
                "definition": "definition_response",
                "default": "default_response"
            }
        )
        
        # Add edges from response nodes to END
        workflow.add_edge("factual_response", END)
        workflow.add_edge("analytical_response", END)
        workflow.add_edge("comparison_response", END)
        workflow.add_edge("definition_response", END)
        workflow.add_edge("default_response", END)
        
        # Compile the graph
        self.graph = workflow.compile()
    
    def _classify_query(self, state: QueryState) -> QueryState:
        """Classify the query into a category.
        
        Args:
            state: Current state containing the question
            
        Returns:
            Updated state with category
        """
        classifier_chain = CLASSIFIER_PROMPT | self.llm | StrOutputParser()
        category = classifier_chain.invoke({"question": state["question"]}).strip().lower()
        
        # Ensure category is valid
        valid_categories = ["factual", "analytical", "comparison", "definition", "default"]
        if category not in valid_categories:
            category = "default"
        
        print(f"Question: {state['question']}, Category: {category}")
        
        return {
            **state,
            "category": category
        }
    
    def _route_query(self, state: QueryState) -> Literal["factual", "analytical", "comparison", "definition", "default"]:
        """Route to the appropriate response node based on category.
        
        Args:
            state: Current state containing the category
            
        Returns:
            Category name to route to
        """
        return state["category"]
    
    def _factual_response(self, state: QueryState) -> QueryState:
        """Generate factual response.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with response
        """
        chain = RESPONSE_PROMPTS["factual"] | self.llm | StrOutputParser()
        response = chain.invoke({"question": state["question"]})
        return {**state, "response": response}
    
    def _analytical_response(self, state: QueryState) -> QueryState:
        """Generate analytical response.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with response
        """
        chain = RESPONSE_PROMPTS["analytical"] | self.llm | StrOutputParser()
        response = chain.invoke({"question": state["question"]})
        return {**state, "response": response}
    
    def _comparison_response(self, state: QueryState) -> QueryState:
        """Generate comparison response.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with response
        """
        chain = RESPONSE_PROMPTS["comparison"] | self.llm | StrOutputParser()
        response = chain.invoke({"question": state["question"]})
        return {**state, "response": response}
    
    def _definition_response(self, state: QueryState) -> QueryState:
        """Generate definition response.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with response
        """
        chain = RESPONSE_PROMPTS["definition"] | self.llm | StrOutputParser()
        response = chain.invoke({"question": state["question"]})
        return {**state, "response": response}
    
    def _default_response(self, state: QueryState) -> QueryState:
        """Generate default response.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with response
        """
        chain = RESPONSE_PROMPTS["default"] | self.llm | StrOutputParser()
        response = chain.invoke({"question": state["question"]})
        return {**state, "response": response}
    
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
        # Initialize state
        initial_state = {
            "question": message,
            "category": "",
            "response": ""
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        # Return the response
        return result["response"]