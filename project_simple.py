from google import genai
import numpy as np
import warnings
import logging


client = genai.Client()

documents = [
    "Refund requests are accepted within 7 days of purchase.",
    "Students must complete all assignments before placement support starts.",
    "Eligible students receive minimum 3 interview opportunities.",
    "Course access is available for 6 months"
]

question = "How many interview oportunitites can I get?"

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

scores = []

for vector in documents_vector:
    score = cosine_similarity(question_vector, vector)
    scores.append(score)

best_index = np.argmax(scores)
best_document = documents[best_index]

print("Retrived Context:")
print(best_document)

prompt = f"""
Answer the questions using only the context below.

Context :
{best_document}

Question:
{question}
"""

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)


response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)

print("\nFinal Answer:")
print(response.text)


