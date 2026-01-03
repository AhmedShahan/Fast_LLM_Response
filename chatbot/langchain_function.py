from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()


def langchainChatbot(temperature, max_output_tokens,query) -> str: 
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    # Ask a question
    response = llm.invoke(query)

    return response.content

if __name__ == "__main__":
    query = "What is AI"
    print(langchainChatbot(0.7, 500, query))