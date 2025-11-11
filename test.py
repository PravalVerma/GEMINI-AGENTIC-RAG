from qdrant_client import QdrantClient
client = QdrantClient(
    url="https://0b365b04-9f10-484f-879a-05fa61714a07.us-east-1-1.aws.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.5urZ6CrxvGfLi3gL4RlUKSl4ygw4ozj9x7iF-nNpBrY",
    timeout=60,
)
print(client.get_allocated_memory())  # simple authenticated call (use a real QdrantClient method)

