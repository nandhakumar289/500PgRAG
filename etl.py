from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import Settings
from llama_index.core.ingestion import IngestionPipeline

from llama_index.embeddings.ollama import OllamaEmbedding
from custom_embedding_model import CustomEmbedding
from transformers import AutoTokenizer, AutoModel



# Specify the path to the folder containing your documents
folder_path = r'source_docs'

# Initialize the SimpleDirectoryReader
reader = SimpleDirectoryReader(folder_path)

# Load documents from the folder
documents = reader.load_data()

# Settings.embed_model = OllamaEmbedding(model_name="dhanaabhirajk/e5-base-v2",request_timeout=120.0, base_url="http://localhost:11434")
tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base-v2")
model = AutoModel.from_pretrained("intfloat/e5-base-v2")

# Create a custom embedding instance
custom_embedding = CustomEmbedding(
    tokenizer, model, "intfloat/e5-base-v2"
)
Settings.embed_model = custom_embedding

# Initialize the Milvus VectorStore
vector_store = MilvusVectorStore(uri="http://localhost:19530", dim = 768, collection_name='source_docs_collection', overwrite=True)

pipeline = IngestionPipeline(
vector_store=vector_store,
)
pipeline.run(documents=documents, show_progress=True)

print("Documents loaded into Milvus vector store successfully!")

