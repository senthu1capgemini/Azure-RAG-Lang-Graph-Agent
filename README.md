**Introduction**
This repository provides a template to create a simple Azure Langchain Langgraph agent workflow that is capable of performing RAG search, web search, summarization, and calculation.
This repository was created via uv and all dependancies can be found on the pyproject.toml file.

**Components**
- LLM : Azure Foundry OpenAI GPT4o
- Embeddings model : Azure Foundry OpenAI Embedding model
- RAG : Azure Search - Vectorizers, Skillsets, Indexers, Indexes
- Persistent Memory : SQLite
- Tools : Calculator, Summarizer, Rag Search, Web Search (Tavily)

Additional Guardrails -> Can be implemented via Azure Foundry on model deployment

**Langchain Agent - Langgraph Workflow**
- Start at "agent" → model makes a decision.
- If tools are needed → go to "tools".
- After tool use → return to "agent" for further reasoning.
- If no further action is needed → end the workflow
