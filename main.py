import os
import operator
import psycopg2
import sqlite3
from dotenv import load_dotenv        
from tavily import TavilyClient

from typing import TypedDict, Annotated, Sequence
from typing_extensions import TypedDict

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery

load_dotenv()
SQLITE_DB_PATH ="chat_history.db"

# conn = sqlite3.connect("chat_history.db")
# cursor = conn.cursor()

# # Create the table if it doesn't exist
# cursor.execute("""
#     CREATE TABLE IF NOT EXISTS chat_messages (
#         session_id TEXT,
#         role TEXT,
#         content TEXT
#     )
# """)

# conn.commit()
# conn.close()



#Azure Components Initialization - LLM, Embeddings, Vector Store, Memory----------------------------------------------------------------------
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
class AzureSearchVector:
    def __init__(self, endpoint, key, index_name, embeddings, vector_field="contentVector", text_field="content"):
        # Clean endpoint
        endpoint = endpoint.rstrip('/')
        if '/indexes/' in endpoint:
            endpoint = endpoint.split('/indexes/')[0]
        
        # Create credential properly
        credential = AzureKeyCredential(key)
        try:
            self.client = SearchClient(
                endpoint=endpoint,
                index_name=index_name,
                credential=credential  
            )
            self.embeddings = embeddings
            self.vector_field = vector_field
            self.text_field = text_field
            
            # Test connection
            print(f"  Testing connection...")
            # Try to get document count
            results = self.client.search(search_text="*", top=1, include_total_count=True)
            print(f"Connected successfully to Azure Search")
            
        except Exception as e:
            print(f"Failed to initialize Azure Search client")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 3):
        try:
            print(f"Performing similarity search for: '{query}'")
            
            # Generate embedding
            query_vector = self.embeddings.embed_query(query)
            print(f"Generated embedding vector of length: {len(query_vector)}")
            
            # Create vector query
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=k,
                fields=self.vector_field
            )
            
            print(f"Searching in field: {self.vector_field}")
            print(f"Returning field: {self.text_field}")
            
            # Search
            results = self.client.search(
                search_text=None,
                vector_queries=[vector_query],
                select=[self.text_field],
                top=k
            )
            
            # Convert to documents
            docs = []
            for i, result in enumerate(results):
                content = result.get(self.text_field, "")
                if content:
                    print(f" Result {i+1}: {content[:100]}...")
                    docs.append(Document(page_content=content))
            
            print(f"Found {len(docs)} results")
            return docs
            
        except Exception as e:
            print(f"Search error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return []
        
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
def get_session_history(session_id: str) -> SQLChatMessageHistory:
    """Get or create SQLite-backed chat history for a session."""
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=f"sqlite:///{SQLITE_DB_PATH}"
    )

def clear_session_history(session_id: str = None):
    """Clear chat history for a specific session or all sessions."""
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor() # Needed to execute the SQL commands
        if session_id:
            # Clear specific session
            cursor.execute("Delete from message_store where session_id: ", (session_id,))
            rows_deleted = cursor.rowcount
            conn.commit()
            conn.close()
            
            if rows_deleted > 0:
                return f"Cleared {rows_deleted} messages for session: {session_id}"
            return f"No history for session: {session_id}"
        else:
            # Clear all sessions
            cursor.execute("Delete from message_store")
            rows_deleted = cursor.rowcount
            conn.commit()
            conn.close()
            return f"Cleared all chat histories ({rows_deleted} messages)"
    except Exception as e:
        return f"Error clearing history: {str(e)}"

def list_sessions():
    """List all available sessions in the database."""
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT session_id, COUNT(*) as message_count, 
                   MIN(id) as first_message_id, MAX(id) as last_message_id
            FROM message_store 
            GROUP BY session_id 
            ORDER BY last_message_id DESC
        """)
        
        sessions = cursor.fetchall()
        conn.close()
        
        if not sessions:
            return "No sessions found in database."
        
        result = "\nAvailable sessions:\n" + "="*50 + "\n"
        for session_id, count, first_id, last_id in sessions:
            result += f"  - {session_id}: {count} messages\n"
        
        return result
    except Exception as e:
        return f"Error listing sessions: {str(e)}"

# Dictionary to store chat histories by session_id-----------------------------------------------------------------------------------------------------------------
chat_histories = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Get or create chat history for a session."""
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]

def clear_session_history(session_id: str = None):
    """Clear chat history for a specific session or all sessions."""
    if session_id:
        if session_id in chat_histories:
            chat_histories[session_id].clear()
            return f"Cleared history for session: {session_id}"
        return f"No history found for session: {session_id}"
    else:
        chat_histories.clear()
        return "Cleared all chat histories"
    
def list_sessions():
    """List all available sessions."""
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()

    # Query all rows from the chat_messages table
    cursor.execute("SELECT session_id, role, content FROM chat_messages")
    rows = cursor.fetchall()

    # Print each row
    for row in rows:
        session_id, role, content = row
        print(f"[{session_id}] {role}: {content}")

    # Close the connection
    conn.close()


#  Define Tools - Mathematical Calculation, Text Summarization, Knowledge Base Search------------------------------------------------------------------------------
@tool
def calculate(expression: str) -> str:
    """Performs mathematical calculations. Input should be a valid Python math expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"

@tool
def summarize_text(text: str) -> str:
    """Summarizes the given text using the LLM."""
    prompt = f"Please provide a concise summary of the following text:\n\n{text}"
    response = llm.invoke(prompt)
    return response.content

@tool
def search_knowledge_base(query: str) -> str:
    """Searches the Azure AI Search vector database for relevant information."""
    results = vector_store.similarity_search(query, k=3)
    if not results:
        return "No relevant information found in the knowledge base."
    
    context = "\n\n".join([doc.page_content for doc in results])
    return f"Found relevant information:\n{context}"

@tool
def web_search(query: str,  num_results: int = 3) -> str:
    """Searches the web using tavily search and provides upto 5 results."""
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        tavily_client = TavilyClient(api_key)
        client = TavilyClient("tvly-dev-********************************")
        response = tavily_client.search(
                query=query,
                max_results=3,
                search_depth="basic"  # or "advanced" for more thorough search
            )
        print(response)
        return response['results']
    except ValueError as e:
        print(f"{e}")


# Create tools list
tools = [calculate, summarize_text, search_knowledge_base, web_search]

# Create ToolNode - this replaces ToolExecutor and individual tool nodes
tool_node = ToolNode(tools)

# LangGraph State Definition--------------------------------------------------------------------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# Define LangGraph workflow nodes
def call_model(state: AgentState):
    """Calls the LLM to decide what to do next."""
    messages = state["messages"]
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Get response
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Simplified routing logic
def should_continue(state: AgentState):
    """Determines if we should continue to tools or end."""
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
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)
workflow.add_edge("tools", "agent")

# Compile the graph
app = workflow.compile()

# Main execution function of agent with memory------------------------------------------------------------------------------------------------------
def run_agent(user_input: str, session_id: str = "default"):
    
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
    """Main interactive CLI loop."""
    print("\n" + "="*70)
    print("   INTERACTIVE AGENT CLI")
    print("="*70)
    print("\nCommands:")
    print("  - Type your query and press Enter to talk to the agent")
    print("  - 'status' - Show agent status")
    print(f"\n📁 Database: {SQLITE_DB_PATH}")
    print("  - 'clear' - Clear current session history")
    print("  - 'sessions' - List all available sessions")
    print("  - 'session <name>' - Switch to a different session")
    print("\nAvailable Tools:")
    print("  - calculate(expression) - Perform math calculations")
    print("  - summarize_text(text) - Summarize long text")
    print("  - search_knowledge_base(query) - Search the knowledge base")
    print("  - web search(query) - Search the web via Tavily")
    print("\n" + "="*70 + "\n")

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
                print("\n Goodbye!\n")
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

            elif user_input.lower() == 'clear':
                result = clear_session_history(current_session)
                print(f"\nCleared session {result}\n")
                continue
            elif user_input.lower() == 'session_history':
                result = get_session_history()
                continue
            
            
            # Run the agent
            print(f"\n[{current_session}] Agent: ", end="", flush=True)
            response = run_agent(user_input, session_id=current_session)
            print(response)
        
        except KeyboardInterrupt:
            print("\n\n Interrupted. Goodbye!\n")
            break
        
        except Exception as e:
            print(f"\n Error: {e}\n")

if __name__ == "__main__":
    interactive_cli()

