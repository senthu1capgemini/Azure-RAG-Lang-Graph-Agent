import os
from langchain_core.tools import tool
from tavily import TavilyClient
from ragSearch import AzureSearchVector
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from dotenv import load_dotenv     

load_dotenv()

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

#  Define Tools - Mathematical Calculation, Text Summarization, Knowledge Base Search------------------------------------------------------------------------------
@tool
def calculate(expression: str) -> str:
    "Performs mathematical calculations. Input should be a valid Python math expression."
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
    "Searches the Azure AI Search vector database for relevant information."
    results = vector_store.similarity_search(query, k=3)
    if not results:
        return "No relevant information found in the knowledge base."
    
    context = "\n\n".join([doc.page_content for doc in results])
    return f"Found relevant information:\n{context}"

@tool
def web_search(query: str,  num_results: int = 3) -> str:
    "Searches the web using tavily search and provides upto 5 results."
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
