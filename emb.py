from google import genai
import numpy as np

client = genai.Client()

documents = [
    "Refund requests are accepted within 7 days of purchase.",
    "Students must complete all assignments before placement support starts.",
    "Eligible students receive minimum 3 interview opportunities."
]

question = "Can I get my money back after 5 days?"

MODEL = "gemini-embedding-2"   # same model for everything

documents_vector = []

for document in documents:
    result = client.models.embed_content(
        model=MODEL,
        contents=document
    )
    vector = result.embeddings[0].values
    documents_vector.append(vector)

question_result = client.models.embed_content(
    model=MODEL,
    contents=question
)

question_vector = question_result.embeddings[0].values

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

for document, vector in zip(documents, documents_vector):
    score = cosine_similarity(question_vector, vector)
    print(document, "->", round(score,3))