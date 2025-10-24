# Perplexia AI - Student Skeleton Code

This is the skeleton codebase for the Perplexia AI project, supporting Week 1, Week 2, and Week 3 assignments.

## Project Structure

```
code/
├── perplexia_ai/
│   ├── core/
│   │   └── chat_interface.py      # Base interface for all implementations
│   ├── tools/
│   │   └── calculator.py          # Calculator tool for Week 1
│   ├── week1/
│   │   ├── factory.py             # Week 1 factory
│   │   ├── part1.py               # Query Understanding
│   │   ├── part2.py               # Basic Tools
│   │   └── part3.py               # Memory
│   ├── week2/
│   │   ├── factory.py             # Week 2 factory
│   │   ├── part1.py               # Web Search (STUB)
│   │   ├── part2.py               # Document RAG (STUB)
│   │   └── part3.py               # Corrective RAG (STUB)
│   ├── week3/
│   │   ├── factory.py             # Week 3 factory
│   │   ├── part1.py               # Tool-Using Agent (STUB)
│   │   ├── part2.py               # Agentic RAG (STUB)
│   │   └── part3.py               # Deep Research (STUB)
│   ├── solutions/
│   │   ├── week1/                 # Week 1 solution code
│   │   ├── week2/                 # Week 2 solution code (will be shared later)
│   │   └── week3/                 # Week 3 solution code (will be shared later)
│   └── app.py                     # Gradio app setup (DO NOT MODIFY)
└── run.py                         # Main entry point (DO NOT MODIFY)
```

## Quick Start

### Running the Application

```bash
# Student Code
python run.py --week 1 --mode part1
python run.py --week 2 --mode part1
python run.py --week 3 --mode part1

# Solution Code (for reference/once released)
python run.py --week 1 --mode part1 --solution
python run.py --week 2 --mode part1 --solution
python run.py --week 3 --mode part1 --solution
```

## What to Implement

### Week 2

#### Part 1: Web Search (`week2/part1.py`)
- Set up Tavily search tool
- Create LangGraph workflow for web search
- Add Opik tracing for observability
- Format responses with citations

#### Part 2: Document RAG (`week2/part2.py`)
- Load and process OPM annual reports
- Create vector embeddings using OpenAI
- Build LangGraph retrieval workflow
- Generate responses with source citations

#### Part 3: Corrective RAG (`week2/part3.py`)
- Implement document relevance grading
- Create conditional routing between document RAG and web search
- Combine multiple knowledge sources
- Handle information conflicts

### Week 3

#### Part 1: Tool-Using Agent (`week3/part1.py`)
- Define tools (calculator, datetime, weather)
- Create ReAct agent using LangGraph
- Implement autonomous tool selection
- Compare with manual workflows

#### Part 2: Agentic RAG (`week3/part2.py`)
- Build agent with document retrieval tools
- Add web search capability
- Implement dynamic strategy selection
- Create autonomous information gathering

#### Part 3: Deep Research (`week3/part3.py`)
- Design multi-agent system
- Create specialized agents (planner, researcher, writer)
- Implement agent coordination
- Generate comprehensive research reports

## Key Patterns

### ChatInterface Implementation

All implementations must inherit from `ChatInterface`:

```python
from perplexia_ai.core.chat_interface import ChatInterface

class MyChat(ChatInterface):
    def __init__(self):
        # Initialize instance variables
        pass
    
    def initialize(self) -> None:
        # Set up LLM, tools, graphs, etc.
        pass
    
    def process_message(self, message: str, chat_history: List[Dict[str, str]]) -> str:
        # Process the message and return response
        return "response"
```

### LangGraph Pattern (Week 2 & 3)

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class MyState(TypedDict):
    query: str
    result: str

graph = StateGraph(MyState)
graph.add_node("process", process_function)
graph.add_edge(START, "process")
graph.add_edge("process", END)
self.graph = graph.compile()
```

## Files You Should NOT Modify

- `app.py` - Gradio interface setup
- `run.py` - Entry point
- `core/chat_interface.py` - Base interface
- `week*/factory.py` - Factory patterns

## Files You SHOULD Modify

- `week1/part*.py` - Your Week 1 implementations
- `week2/part*.py` - Your Week 2 implementations
- `week3/part*.py` - Your Week 3 implementations

You can create your own prompt files wherever you prefer to organize them.

## Helpful Resources

### Environment Setup
Make sure you have:
- OpenAI API key in `.env` file
- Tavily API key (for Week 2 & 3)
- Required packages: `langchain`, `langgraph`, `gradio`, `openai`, `tavily-python`, `opik`

## Testing Your Implementation

1. Replace the stub implementation in the appropriate `part*.py` file
2. Run the application using `run.py`
3. Test using the Gradio web interface
4. Check console output for debugging information

## Tips

1. **Start Simple**: Get basic functionality working before adding complexity
2. **Use Prompts**: Reference `week2/prompts.py` for examples
3. **Debug Incrementally**: Test each node in your graph separately
4. **Check Solution Code**: Week 1 solutions are available as reference
5. **Read Docstrings**: Each stub has detailed instructions in docstrings

## Support

- Check the `CHANGES.md` file for detailed information about the codebase structure
- Review Week 1 solution code for implementation patterns
- Refer to LangChain and LangGraph documentation

## Good Luck! 🚀

Remember: The infrastructure is already set up. You just need to focus on implementing the LangChain/LangGraph logic!
