from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from langchain_core.documents import Document

class AzureSearchVector:
    def __init__(self, endpoint, key, index_name, embeddings, vector_field="contentVector", text_field="content"):
        endpoint = endpoint.rstrip('/')
        if '/indexes/' in endpoint:
            endpoint = endpoint.split('/indexes/')[0]
        
        #Create credential properly
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
            
            #Test connection
            print(f"  Testing connection...")
            #get document count
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
            #Generate embedding
            query_vector = self.embeddings.embed_query(query)
            print(f"Generated embedding vector of length: {len(query_vector)}")
            #Create vector query
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=k,
                fields=self.vector_field
            )
            print(f"Searching in field: {self.vector_field}")
            print(f"Returning field: {self.text_field}")
            #Search
            results = self.client.search(
                search_text=None,
                vector_queries=[vector_query],
                select=[self.text_field],
                top=k
            )
            #Convert to documents
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
