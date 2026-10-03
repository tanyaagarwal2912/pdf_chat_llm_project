from google import genai

##creating object
client = genai.Client()


##api call : client -> gemini
stream = client.interactions.create(
    model="gemini-3.6-flash",
    input="Explain machine learning in one simple sentence.give me 10 paragraphs with eg.",
    stream = True
)

for event in stream:
    if event.event_type == "step.delta":
        if event.delta.type == "text":
            print(event.delta.text, end="", flush = True)