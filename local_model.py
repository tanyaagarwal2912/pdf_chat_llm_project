from ollama import chat


response = chat(
    model = "gemma3:1b",
    messages = [
        {
            "role":"user",
            "content":"Explain machine learning in one sentence."
        }
    ]
)
print(response.message.content)