from getpass import getpass
import os
import cohere

api_key = os.getenv("COHERE_API_KEY") or getpass("Cohere API key: ")
client = cohere.ClientV2(api_key=api_key.strip(), timeout=90)
response = client.chat(
    model=os.getenv("COHERE_MODEL", "command-a-03-2025"),
    messages=[{
        "role": "user",
        "content": "Одним реченням українською поясни, що таке промпт.",
    }],
    temperature=0.2,
    max_tokens=1200,
)
for block in response.message.content or []:
    if block.type == "text":
        print(block.text)
print("finish_reason:", response.finish_reason)
