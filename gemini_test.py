from google import genai

##creating object
client = genai.Client()


##api call : client -> gemini
interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input="give me names of 3 top ai startups",
    generation_config = {
        "temperature":1.0,
        "max_output_tokens":200
    }
)

print(interaction.output_text)