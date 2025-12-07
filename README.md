# Introduction

- This repository provides a template to create a simple Azure Langchain Langgraph agent workflow that is capable of performing RAG search, web search, summarization, and calculation.
- This repository was created via uv and all dependancies can be found on the pyproject.toml file.

# Functional Architecture
                ┌───────────────────────────┐
                │        User Query         │
                └─────────────┬─────────────┘
                              │
                              ▼
                ┌───────────────────────────┐
                │        LLM Layer          │
                │  - Azure Chat OpenAI      │
                └─────────────┬─────────────┘
                              │
          ┌───────────────────────┼──────────────────────────────┐
          │                       │                              │
          ▼                       ▼                              ▼
          ┌───────────────────┐ ┌──────────────────┐ ┌───────────────────┐
          │ OpenAI ADA        │ │ Persistent Memory│ │ Tools             │
          │ Embeddings        │ │ SQLite (Local)   │ │ - Calculator      │
          └─────────┬─────────┘ └────────┬─────────┘ │ - Summarizer      │
                    │                    │           │ - Rag Search      │
                    ▼                    │           │ - Web Search      │
         ┌───────────────────┐           │           │   (Tavily)        │
         │ Azure AI Search   │           │           └───────────────────┘
         └─────────┬─────────┘           │
                   │                     │
                   └──────────┬──────────┘
                              │
                              ▼
                ┌───────────────────────────┐
                │   Response Generation     │
                └─────────────┬─────────────┘
                              │
                              ▼
                ┌───────────────────────────┐
                │ Monitoring: Arize Phoenix │
                │ (Local Setup)             │
                └───────────────────────────┘





# Components
- LLM : Azure Foundry OpenAI GPT4o
- Embeddings model : Azure Foundry OpenAI Embedding model
- RAG : Azure Search - Vectorizers, Skillsets, Indexers, Indexes
- Persistent Memory : SQLite
- Tools : Calculator, Summarizer, Rag Search, Web Search (Tavily)

# Azure Resources used 
- Azure Blob Storage - Store raw documents
- Azure Search
- Azure Foundry (LLM and embeddings)


Additional Guardrails -> Can be implemented via Azure Foundry on model deployment

# Langchain Agent - Langgraph Workflow
This is a cyclical loop where the agent decides if/which tool should be used based on the query. It falls back to the agent if further steps are needed.
- Start at "agent" and model makes a decision.
- If tools are needed, go to "tools".
- After tool use, return to "agent" for further reasoning.
- If no further action is needed, end the workflow

# Python Files
There are four files used here.
- main.py : This is the main execution file
- memory.py : This handles all fucntions related to persistent memory via SQLite
- toolkit.py : This handles all tool creation.
- ragSearch.py : This handles the vector database and search functions for Azure Search.

# How to use
- Create the Azure components : Azure Search, Azure Foundry Emdeddings, and LLM via Forundry. Get the API details and save as an env file.
- Run the main.py file
- For Web Search, I use Tavily, you may need to set up an API access for it.
- For monitoring, please use the http://localhost:6006/projects to view token usage and costs of each prompt and response. Additional annotations can be added.
- To view the persistent memory database file .db, please use https://inloop.github.io/sqlite-viewer/.

# Next Steps:
- If necessary, change SQLITE to Azure CosmosDB
- Include Structured RAG (pgvector), or GraphRAG
- Include monitoring via LangFuse, LangSmith
- Include guardrail implementation via Azure Foundry (This might include PII, toxicitiy, adulterated content, deepfakes, etc.) Include Azure Content Safety.
- Continue model experimentation for LLMs and embeddings
- Continue improved indexing via skillsets, indexers, indexes on Azure Search for Retrieval Quality and Knowledge management
- Include a frontened app via StreamLit or Gradio --or better React or Next
- Deploy on Azure Logic Apps for fault tolerance and availability on cloud
- Improved prompting - I did not find this necessary as it really depends on data quality as well.
- Include a means to login via Entra ID for Azure Logic Apps (Last step)















