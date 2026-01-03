from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()  # Load the API key

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5
)

# Ask a question
response = llm.invoke("Write a short story about a cat in a futuristic city.")
print(response.content)