"""Part 3 - Conversation Memory implementation using LangGraph.

This implementation focuses on:
- Maintain context across messages
- Handle follow-up questions
- Use conversation history in responses
"""

import contextlib
import io
from typing import Dict, List, Optional, TypedDict, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END

from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.tools.calculator import Calculator

# Classifier prompt with history support
CLASSIFIER_PROMPT = ChatPromptTemplate.from_template("""
Classify the given user question into one of the specified categories based on its nature, including all defined categories.

- Factual Questions: Questions starting with phrases like "What is...?" or "Who invented...?" should be classified as 'factual'.
- Analytical Questions: Questions starting with phrases like "How does...?" or "Why do...?" should be classified as 'analytical'.
- Comparison Questions: Questions starting with phrases like "What's the difference between...?" should be classified as 'comparison'.
- Definition Requests: Questions starting with phrases like "Define..." or "Explain..." should be classified as 'definition'.
- Datetime Questions: Questions related to date or time computation should be classified as 'datetime'.
- Calculation Questions: Questions requiring mathematical computation, not associated with date or time, should be classified as 'calculation'.

If the question does not fit into any of these categories, return 'default'.

# Steps

1. Analyze the user question.
2. Determine which category the question fits into based on its structure and keywords.
3. Return the corresponding category or 'default' if none apply.

# Output Format

- Return only the category word: 'factual', 'analytical', 'comparison', 'definition', 'datetime', 'calculation', or 'default'.
- Do not include any extra text or quotes in the output.

# Examples

- **Example 1**  
  * Question: What is the highest mountain in the world?  
  * Response: factual

- **Example 2**  
  * Question: What's the difference between OpenAI and Anthropic?  
  * Response: comparison

- **Example 3**  
  * Question: What's an 18% tip of a $105 bill?  
  * Response: calculation

- **Example 4**  
  * Question: What day is it today?  
  * Response: datetime

Use information from the conversation history only if relevant to the above user query, otherwise ignore the history.
Conversation history with the user:
{history}

User question: {question}

""")

# Response prompts for each category (all with history support)
RESPONSE_PROMPTS = {
    "factual": ChatPromptTemplate.from_template(
        """
        Answer the following question concisely with a direct fact. Avoid unnecessary details.

        Use information from the conversation history only if relevant to the above user query, otherwise ignore the history.
        Conversation history with the user:
        {history}

        User question: "{question}"
        Answer:
        """
    ),
    "analytical": ChatPromptTemplate.from_template(
        """
        Provide a detailed explanation with reasoning for the following question. Break down the response into logical steps.

        Use information from the conversation history only if relevant to the above user query, otherwise ignore the history.
        Conversation history with the user:
        {history}

        User question: "{question}"
        Explanation:
        """
    ),
    "comparison": ChatPromptTemplate.from_template(
        """
        Compare the following concepts. Present the answer in a structured format using bullet points or a table for clarity.

        Use information from the conversation history only if relevant to the above user query, otherwise ignore the history.
        Conversation history with the user:
        {history}

        User question: "{question}"
        Comparison:
        """
    ),
    "definition": ChatPromptTemplate.from_template(
        """
        Define the following term and provide relevant examples and use cases for better understanding.

        Use information from the conversation history only if relevant to the above user query, otherwise ignore the history.
        Conversation history with the user:
        {history}

        User question: "{question}"
        Definition:
        Examples:
        Use Cases:
        """
    ),
    "calculation": ChatPromptTemplate.from_template(
        """
        You are a smart AI model but cannot do any complex calculations. You are very good at
        translating a math question to a simple equation which can be solved by a calculator.

        Convert the user question below to a math calculation.
        Remember that the calculator can only use +, -, *, /, //, % operators,
        so only use those operators and output the final math equation.

        Examples:
        Question: What is 5 times 20?
        Answer: 5 * 20

        Question: What is the split of each person for a 4 person dinner of $100 with 20% tip?
        Answer: (100 + 0.2*100) / 4

        Question: Round 100.5 to the nearest integer.
        Answer: 100.5 // 1

        Use information from the conversation history only if relevant to the above user query, otherwise ignore the history.
        Conversation history with the user:
        {history}

        User Query: "{question}"

        The final output should ONLY contain the valid math equation, no words or any other text.
        Otherwise the calculator tool will error out.
        """
    ),
    "datetime": ChatPromptTemplate.from_template(
        """You are a smart AI which is very good at translating a question in english
        to a simple python code to output the result. You'll only be given queries related
        to date and time, for which generate the python code required to get the answer.
        Your code will be sent to a Python interpreter and the expectation is to print the output on the final line.

        These are the ONLY python libraries you have access to - math, datetime, time.

        Examples:
        Question: What day is it today?
        Answer: print(datetime.now().strftime("%A"))

        Question: What is the date of 30 days from now?
        Answer: print(datetime.now() + timedelta(days=30))

        Use information from the conversation history only if relevant to the above user query, otherwise ignore the history.
        Conversation history with the user:
        {history}

        User Query: "{question}"

        The final output should ONLY contain valid Python code, no words or any other text.
        Otherwise the Python interpreter tool will error out. Avoid returning ``` or python
        in the output, just return the code directly.
        """
    ),
    "default": ChatPromptTemplate.from_template(
        """
        Respond your best to answer the following question but keep it very brief.

        Use information from the conversation history only if relevant to the above user query, otherwise ignore the history.
        Conversation history with the user:
        {history}

        User question: "{question}"
        Answer:
        """
    )
}

# Tool decorators for calculator and datetime
@tool
def calculator_tool(expression: str) -> str:
    """Evaluate a math expression and return the result as a string.

    Supports only basic arithmetic operations (+, -, *, /, //, %) and parentheses.

    Args:
        expression: The math expression to evaluate.

    Returns:
        The result of the math expression as a string formatted as "The answer is: {result}"
    """
    print(f"Evaluating expression: {expression}")
    result = str(Calculator.evaluate_expression(expression))
    return f"The answer is: {result}"

@tool
def datetime_tool(code: str) -> str:
    """Execute Python code to answer date or time related questions.
    
    NOTE: We are using exec here to execute the code, which is not a good practice for production
    as this can lead to security vulnerabilities. For the purpose of the assignment, we are assuming
    the model will only return valid and safe python code.

    Args:
        code: The python code to execute.

    Returns:
        The output of the python code as a string.
    """
    print(f"Executing code: {code}")
    output_buffer = io.StringIO()
    code = f"import datetime\nimport time\nfrom datetime import timedelta\n{code}"
    with contextlib.redirect_stdout(output_buffer):
        exec(code)
    return output_buffer.getvalue().strip()

# State definition for the graph
class MemoryState(TypedDict):
    """State for the memory-enabled chat graph."""
    question: str
    history: str
    category: str
    response: str

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
        # Initialize the LLM
        self.llm = init_chat_model("gpt-5-mini", model_provider="openai", reasoning_effort='minimal')
        
        # Build the graph
        self.graph = None
        self._build_graph()
    
    def _build_graph(self) -> None:
        """Build the LangGraph with memory support."""
        # Create the graph
        workflow = StateGraph(MemoryState)
        
        # Add classifier node
        workflow.add_node("classifier", self._classify_query)
        
        # Add response nodes for all categories
        workflow.add_node("factual_response", self._factual_response)
        workflow.add_node("analytical_response", self._analytical_response)
        workflow.add_node("comparison_response", self._comparison_response)
        workflow.add_node("definition_response", self._definition_response)
        workflow.add_node("calculation_response", self._calculation_response)
        workflow.add_node("datetime_response", self._datetime_response)
        workflow.add_node("default_response", self._default_response)
        
        # Set entry point
        workflow.set_entry_point("classifier")
        
        # Add conditional edges from classifier
        workflow.add_conditional_edges(
            "classifier",
            self._route_query,
            {
                "factual": "factual_response",
                "analytical": "analytical_response",
                "comparison": "comparison_response",
                "definition": "definition_response",
                "calculation": "calculation_response",
                "datetime": "datetime_response",
                "default": "default_response"
            }
        )
        
        # Add edges from response nodes to END
        workflow.add_edge("factual_response", END)
        workflow.add_edge("analytical_response", END)
        workflow.add_edge("comparison_response", END)
        workflow.add_edge("definition_response", END)
        workflow.add_edge("calculation_response", END)
        workflow.add_edge("datetime_response", END)
        workflow.add_edge("default_response", END)
        
        # Compile the graph
        self.graph = workflow.compile()
    
    def _classify_query(self, state: MemoryState) -> MemoryState:
        """Classify the query into a category using history."""
        classifier_chain = CLASSIFIER_PROMPT | self.llm | StrOutputParser()
        category = classifier_chain.invoke({
            "question": state["question"],
            "history": state["history"]
        }).strip().lower()
        
        # Ensure category is valid
        valid_categories = ["factual", "analytical", "comparison", "definition", "calculation", "datetime", "default"]
        if category not in valid_categories:
            category = "default"
        
        print(f"Question: {state['question']}, Category: {category}")
        
        return {
            **state,
            "category": category
        }
    
    def _route_query(self, state: MemoryState) -> Literal["factual", "analytical", "comparison", "definition", "calculation", "datetime", "default"]:
        """Route to the appropriate node based on category."""
        return state["category"]
    
    def _factual_response(self, state: MemoryState) -> MemoryState:
        """Generate factual response with history."""
        chain = RESPONSE_PROMPTS["factual"] | self.llm | StrOutputParser()
        response = chain.invoke({
            "question": state["question"],
            "history": state["history"]
        })
        return {**state, "response": response}
    
    def _analytical_response(self, state: MemoryState) -> MemoryState:
        """Generate analytical response with history."""
        chain = RESPONSE_PROMPTS["analytical"] | self.llm | StrOutputParser()
        response = chain.invoke({
            "question": state["question"],
            "history": state["history"]
        })
        return {**state, "response": response}
    
    def _comparison_response(self, state: MemoryState) -> MemoryState:
        """Generate comparison response with history."""
        chain = RESPONSE_PROMPTS["comparison"] | self.llm | StrOutputParser()
        response = chain.invoke({
            "question": state["question"],
            "history": state["history"]
        })
        return {**state, "response": response}
    
    def _definition_response(self, state: MemoryState) -> MemoryState:
        """Generate definition response with history."""
        chain = RESPONSE_PROMPTS["definition"] | self.llm | StrOutputParser()
        response = chain.invoke({
            "question": state["question"],
            "history": state["history"]
        })
        return {**state, "response": response}
    
    def _default_response(self, state: MemoryState) -> MemoryState:
        """Generate default response with history."""
        chain = RESPONSE_PROMPTS["default"] | self.llm | StrOutputParser()
        response = chain.invoke({
            "question": state["question"],
            "history": state["history"]
        })
        return {**state, "response": response}
    
    def _calculation_response(self, state: MemoryState) -> MemoryState:
        """Generate math expression and execute calculator using chained tools with history."""
        # Chain: LLM generates expression -> StrOutputParser -> calculator_tool executes it
        chain = RESPONSE_PROMPTS["calculation"] | self.llm | StrOutputParser() | calculator_tool
        response = chain.invoke({
            "question": state["question"],
            "history": state["history"]
        })
        
        return {**state, "response": response}
    
    def _datetime_response(self, state: MemoryState) -> MemoryState:
        """Generate Python code and execute it using chained tools with history."""
        # Chain: LLM generates code -> StrOutputParser -> datetime_tool executes it
        chain = RESPONSE_PROMPTS["datetime"] | self.llm | StrOutputParser() | datetime_tool
        response = chain.invoke({
            "question": state["question"],
            "history": state["history"]
        })
        
        return {**state, "response": response}
    
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
        # Format chat history as a string
        if chat_history:
            history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
        else:
            history = ""
        
        # Initialize state
        initial_state = {
            "question": message,
            "history": history,
            "category": "",
            "response": ""
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        # Return the response
        return result["response"]
