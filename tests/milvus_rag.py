from pymilvus import MilvusClient

from tests.embedding import emb_text

milvus_client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")

collection_name = "my_rag_collection"

question = "How is data stored in milvus?"

search_res = milvus_client.search(
    collection_name=collection_name,
    data=[
        emb_text(question)
    ],  # Use the `emb_text` function to convert the question to an embedding vector
    limit=3,  # Return top 3 results
    search_params={"metric_type": "IP", "params": {}},  # Inner product distance
    output_fields=["text"],  # Return the text field
)

import json

retrieved_lines_with_distances = [
    (res["entity"]["text"], res["distance"]) for res in search_res[0]
]
print(json.dumps(retrieved_lines_with_distances, indent=4))
