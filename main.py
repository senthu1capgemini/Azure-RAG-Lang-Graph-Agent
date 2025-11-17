import os
import operator
import sqlite3
from dotenv import load_dotenv      

from typing import TypedDict, Annotated, Sequence
from typing_extensions import TypedDict

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from memory import get_session_history, clear_session_history, list_sessions
from toolkit import calculate, summarize_text, search_knowledge_base, web_search
from ragSearch import AzureSearchVector


load_dotenv()
SQLITE_DB_PATH ="chat_history.db"

#Azure Components Initialization - LLM, Embeddings, Vector Store, Memory---------------------------------------------------------------------------
# Initialize Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature = 0.7
)

# Initialize Azure OpenAI Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint = os.getenv("AZURE_EMBEDDINGS_ENDPOINT"),
    api_key = os.getenv("AZURE_EMBEDDINGS_API_KEY"),
    azure_deployment = os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT"),
    api_version = os.getenv("AZURE_EMBEDDINGS_API_VERSION")
)

# Simple Azure Search wrapper that works directly with Azure SDK------------------------------------------------------------------------------------
        
# Initialize vector store
vector_store = AzureSearchVector(
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME"),
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT"),
    key = os.getenv("AZURE_SEARCH_KEY"),
    embeddings=embeddings,
    vector_field="text_vector",
    text_field="chunk",  
)

# SQLite-based chat history management-----------------------------------------------------------------------------------------------------------------
chat_histories = {}

# Create tools list
tools = [calculate, summarize_text, search_knowledge_base, web_search]

# Create ToolNode - this replaces ToolExecutor and individual tool nodes
tool_node = ToolNode(tools)

# LangGraph State Definition----------------------------------------------------------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# Define LangGraph workflow nodes
def call_model(state: AgentState):
    "Calls the LLM to decide what to do next."
    messages = state["messages"]
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Get response
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Simplified routing logic
def should_continue(state: AgentState):
    "Determines if we should continue to tools or end."
    last_message = state["messages"][-1]
    
    # Check if the LLM made a tool call
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    return "end"

# Build LangGraph workflow
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)  # Single ToolNode handles all tools

# Set entry point
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue,
    { 
      "tools": "tools",
      "end": END
    }
)
workflow.add_edge("tools", "agent")

# Compile the graph
app = workflow.compile()
# Main execution function of agent with memory------------------------------------------------------------------------------------------------------
def run_agent(user_input: str, session_id: str = "defaultUser"):
    # Get conversation history
    chat_history = get_session_history(session_id)
    # Load previous messages
    previous_messages = chat_history.messages
    # Create initial state
    initial_state = {
        "messages": previous_messages + [HumanMessage(content=user_input)]
    }
    # Run the workflow
    result = app.invoke(initial_state)
    # Save to memory
    chat_history.add_user_message(user_input)
    # Get the final AI message
    final_message = result["messages"][-1]
    if hasattr(final_message, 'content'):
        chat_history.add_ai_message(final_message.content)
        return final_message.content
    
    return str(final_message)

# ====================================================================================================================================================================================================
# INTERACTIVE CLI INTERFACE
# ====================================================================================================================================================================================================

def interactive_cli():
    "Main interactive CLI loop."
    print("\n" + "="*100)
    print("   INTERACTIVE AGENT CLI")
    print("="*100)
    print("\nCommands:")
    print("  - Type your query and press Enter to talk to the agent")
    print("  - 'status' - Show agent status")
    print("  - 'clear' - Clear current session history")
    print("  - 'sessions' - List all available sessions")
    print("  - 'session <name>' - Switch to a different session")
    print("\nAvailable Tools:")
    print("  - calculate(expression) - Perform math calculations")
    print("  - summarize_text(text) - Summarize long text")
    print("  - search knowledge base(query) - Search the knowledge base")
    print("  - web search(query) - Search the web via Tavily")
    print("\n" + "="*100 + "\n")

    print("Enter your username to start:")
    username = input(f"").strip()
    current_session = username
    print(f"Starting session: {current_session}")

    while True:
        try:
            # Get user input
            user_input = input(f"\n[{current_session}] You: ").strip()
            
            # Handle empty input
            if not user_input:
                continue    
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n Goodbye! Your are on your own now!\n")
                break          
            elif user_input.lower() == 'clear':
                if clear_session_history(current_session):
                    print(f"\n Cleared history for session '{current_session}'\n")
                else:
                    print(f"\n No history to clear for session '{current_session}'\n")
                continue          
            elif user_input.lower().startswith('session '):
                new_session = user_input.split(' ', 1)[1].strip()
                if new_session:
                    current_session = new_session
                    print(f"\n Switched to session: {current_session}\n")
                else:
                    print("\n Please provide a session name\n")
                continue
            elif user_input.lower() == 'sessions':
                result = get_session_history()
                continue
            
            # Run the agent
            print(f"\n[{current_session}] Agent: ", end="", flush=True)
            response = run_agent(user_input, session_id=current_session)
            print(response)
        
        except KeyboardInterrupt:
            print("\n\n Interrupted. Goodbye! Stupid of me or beyond my control!\n")
            break
        
        except Exception as e:
            print(f"\n Error: {e}\n")

if __name__ == "__main__":
    interactive_cli()

