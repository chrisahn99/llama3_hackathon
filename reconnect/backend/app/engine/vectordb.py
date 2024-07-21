import os
from llama_index.vector_stores.milvus import MilvusVectorStore


def get_vector_store():
    address = os.getenv("MILVUS_ADDRESS")
    collection = os.getenv("MILVUS_COLLECTION")
    if not address or not collection:
        raise ValueError(
            "Please set MILVUS_ADDRESS and MILVUS_COLLECTION to your environment variables"
            " or config them in the .env file"
        )
    store = MilvusVectorStore(
        uri=address,
        token=os.getenv("MILVUS_TOKEN"),
        collection_name=collection,
        dim=int(os.getenv("EMBEDDING_DIM")),
    )
    return store
