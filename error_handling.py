from google import genai
from google.genai import errors

##creating object
client = genai.Client()


##api call : client -> gemini
try:
    response = client.interactions.create(
        model="gemini-3.6-flash",
        input="explain python",
    )
    print(response.text)

except errors.APIError as e:
    print("Gemini API error", e.code)
    print(e.message)