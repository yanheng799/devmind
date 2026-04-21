
from openai import OpenAI

input_text = "衣服的质量杠杠的"

client = OpenAI(
    # api_key="sk-d81771b2c8f94ab48b2a884e226eea87",
    api_key="sk-901f743555d246408a9492dc96d57caa",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# completion = client.embeddings.create(
#     model="text-embedding-v4",
#     input=input_text
# )
#
# print(completion.model_dump_json())



def emb_text(text):
    return (
        client.embeddings.create(input=text, model="text-embedding-v4")
        .data[0]
        .embedding
    )

print(emb_text(input_text))

