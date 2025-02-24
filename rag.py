import os
from llama_index.llms.openai import OpenAI

# Load API key from environment variable
os.environ["OPENAI_API_KEY"] = ""
# Initialize OpenAI LLM
llm = OpenAI(model="gpt-4o")

# ------------------------------------------------------------------------------------#

from llama_index.vector_stores.milvus import MilvusVectorStore

collection_name = "source_docs_collection"
user_query = "Does this support I2C?"
def get_vector_store(collection_name) -> MilvusVectorStore:

    return MilvusVectorStore(
        uri="http://localhost:19530",
        dim=768,
        collection_name=collection_name,
    )
    
from llama_index.core import Settings
from custom_embedding_model import CustomEmbedding
from transformers import AutoTokenizer, AutoModel

model_name = "intfloat/e5-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModel.from_pretrained(model_name)
Settings.embed_model = CustomEmbedding(
    model_name=model_name, model=model, tokenizer=tokenizer
)

from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_vector_store(
    vector_store=get_vector_store(collection_name=collection_name)
)

# Create a search query
def create_search_query(user_query:str):
    print("Generating search query...")
    SEARCH_QUERY_PROMPT = """
    You act as a search query generator for the given user query.
    Please generate a search query for the following user query:
    {user_query}
    """
    search_query = llm.complete(SEARCH_QUERY_PROMPT.format(user_query=user_query))
    return search_query.text

# Retrieve relevant text
def relevant_text_retriever(search_query):
    print("Retrieving relevant text...")
    retriever = index.as_retriever(similarity_top_k=30)
    retrieved_nodes = retriever.retrieve(search_query)
    return retrieved_nodes

# Retrieve and frame the context
def context_retriever(retrieved_nodes):
    print("Retrieving context...")
    context = ""
    for node in retrieved_nodes:
        context += node.text + "\n"
    return context

# synthesize the response
SYNTHESIS_PROMPT = """
You act as a response synthesizer for the given user query.
Please synthesize a response for the following user query and context: 
User Query: {user_query}
Context: {context}
"""
def response_synthesizer(user_query, context):
    print("Synthesizing response...")
    response = llm.complete(SYNTHESIS_PROMPT.format(user_query=user_query, context=context))    
    return response

# Orchestrator
def orchestrator(user_query):
    search_query = create_search_query(user_query)
    retrieved_nodes = relevant_text_retriever(search_query)
    context = context_retriever(retrieved_nodes)
    response = response_synthesizer(user_query, context)
    print(response)

def main():
    orchestrator(user_query)

if __name__ == "__main__":
    main()

