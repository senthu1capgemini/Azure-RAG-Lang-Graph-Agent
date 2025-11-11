import os
import operator
from dotenv import load_dotenv

from typing import TypedDict, Annotated, Sequence
from typing_extensions import TypedDict

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery

load_dotenv()

# Environment variables setup
AZURE_OPENAI_ENDPOINT = AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY = AZURE_OPENAI_API_KEY
AZURE_OPENAI_DEPLOYMENT = AZURE_OPENAI_DEPLOYMENT
AZURE_OPENAI_API_VERSION = AZURE_OPENAI_API_VERSION
AZURE_EMBEDDINGS_ENDPOINT = AZURE_EMBEDDINGS_ENDPOINT
AZURE_EMBEDDINGS_API_KEY = AZURE_EMBEDDINGS_API_KEY
AZURE_EMBEDDINGS_DEPLOYMENT = AZURE_EMBEDDINGS_DEPLOYMENT
AZURE_EMBEDDINGS_API_VERSION = AZURE_EMBEDDINGS_API_VERSION
AZURE_SEARCH_INDEX_NAME = AZURE_SEARCH_INDEX_NAME
AZURE_SEARCH_ENDPOINT = AZURE_SEARCH_ENDPOINT
AZURE_SEARCH_KEY = AZURE_SEARCH_KEY


#Azure Components Initialization - LLM, Embeddings, Vector Store, Memory----------------------------------------------------------------------
# Initialize Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    api_version = AZURE_OPENAI_API_VERSION,
    temperature=0.7
)

# Initialize Azure OpenAI Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint = AZURE_EMBEDDINGS_ENDPOINT,
    api_key = AZURE_EMBEDDINGS_API_KEY ,
    azure_deployment = AZURE_EMBEDDINGS_DEPLOYMENT,
    api_version = AZURE_EMBEDDINGS_API_VERSION
)

# Simple Azure Search wrapper that works directly with Azure SDK----------------------------------------------------------------------
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
                credential=credential  # Pass the credential object, not the string
            )
            self.embeddings = embeddings
            self.vector_field = vector_field
            self.text_field = text_field
            
            # Test connection
            print(f"  Testing connection...")
            # Try to get document count
            results = self.client.search(search_text="*", top=1, include_total_count=True)
            print(f"✓ Connected successfully to Azure Search")
            
        except Exception as e:
            print(f"✗ Failed to initialize Azure Search client")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error message: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 3):
        try:
            print(f"\nPerforming similarity search for: '{query}'")
            
            # Generate embedding
            query_vector = self.embeddings.embed_query(query)
            print(f"  Generated embedding vector of length: {len(query_vector)}")
            
            # Create vector query
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=k,
                fields=self.vector_field
            )
            
            print(f"  Searching in field: {self.vector_field}")
            print(f"  Returning field: {self.text_field}")
            
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
                    print(f"  Result {i+1}: {content[:100]}...")
                    docs.append(Document(page_content=content))
            
            print(f"✓ Found {len(docs)} results")
            return docs
            
        except Exception as e:
            print(f"✗ Search error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return []
        
# # Initialize vector store
vector_store = AzureSearchVector(
    index_name = AZURE_SEARCH_INDEX_NAME,
    endpoint = AZURE_SEARCH_ENDPOINT,
    key = AZURE_SEARCH_KEY,
    embeddings=embeddings,
    vector_field="text_vector",
    text_field="chunk",  
)

# Dictionary to store chat histories by session_id
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

#  Define Tools using @tool decorator (recommended modern approach)
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

# Create tools list
tools = [calculate, summarize_text, search_knowledge_base]

# Create ToolNode - this replaces ToolExecutor and individual tool nodes
tool_node = ToolNode(tools)

# LangGraph State Definition
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

# Add nodes - only need agent and tools now
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)  # Single ToolNode handles all tools

# Set entry point
workflow.set_entry_point("agent")

# Add conditional routing from agent
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

# Tool node automatically returns to agent
workflow.add_edge("tools", "agent")

# Compile the graph
app = workflow.compile()

# Main execution function
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


if __name__ == "__main__":

    # Run the agent
    print("\n--- Running Agent ---")
    response = run_agent("What is 15 * 3?", session_id="userSenthu")
    print(f"Agent: {response}")
    
    print("\n--- Running Agent with Knowledge Base ---")
    response = run_agent("Can you tell me about the AI healthcare trends as per CSIRO? Please use Knowledge base.", session_id="userSenthu")
    print(f"Agent: {response}")
    
    print("\n--- Running Agent with Summarization ---")
    long_text = """Expand and diversify retraining pathways: Enhance VET and short-course offerings to quickly adapt to changing skills demands across AI jobs and ensure they remain industry-relevant. This should include expanding existing offerings that have proven effective, co-designing new pathways or offerings with industry, embedding industry credentials / training where appropriate, and implementing Modern Digital Apprenticeship programs at federal and state levels. Diversifying pathways is particularly important for people retraining mid career into areas with greater expected shortages and larger skills changes such as Engineering and will be instrumental in improving diversity in the tech workforce.
    Promote awareness of AI jobs and skills needs: While it is essential to understand and address the potential impacts of AI on current occupations, we need to have an equal focus on ensuring domestic supply of AI-skilled workers meets demand in the economy to avoid future labour market shortages. This requires us to enhance awareness of the job opportunities AI creates and support Australians to understand training and career pathways. Promote AI literacy across the workforce: Support widespread training initiatives to boost AI literacy across the workforce, ensuring that the workforce is prepared for a future where AI adoption is widespread. This needs to include action to upskill senior management in AI governance and adoption.
    """
    response = run_agent(f"Summarize this: {long_text}", session_id="userSenthu")
    print(f"Agent: {response}")