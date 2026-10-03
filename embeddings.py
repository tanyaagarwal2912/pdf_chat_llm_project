## meaning based searching

from google import genai

client = genai.Client()

result = client.models.embed_content(
   model="gemini-embedding-2",
   contents="Refund requests are allowed within 7 days." 
)
#print(result.embeddings)


vector = result.embeddings[0].values
print(vector)