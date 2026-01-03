from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from functools import partial
from dotenv import load_dotenv
load_dotenv()

# Define the state type
class ChatAttribute(TypedDict):
    AI_message: str
    human_message: str

# The chatbot function
def langGraphChatbot(state: ChatAttribute, query: str, temperature: float, max_output_tokens: int) -> ChatAttribute:
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=temperature,
        max_output_tokens=max_output_tokens
    )

    # Chain with output parser
    parser = StrOutputParser()
    chain = llm | parser

    # Invoke LLM
    response = chain.invoke(query)

    # Store AI response in state
    state['AI_message'] = response
    state['human_message'] = query  # optional, keep track of query
    return state

# Main function: pass everything dynamically
def main(query: str, temperature: float = 0.7, max_output_tokens: int = 500):
    # Create StateGraph
    graph = StateGraph(ChatAttribute)

    # Wrap the chatbot function so StateGraph only passes 'state'
    node_function = partial(
        langGraphChatbot,
        query=query,
        temperature=temperature,
        max_output_tokens=max_output_tokens
    )

    # Add node and edges
    graph.add_node("chatting", node_function)
    graph.add_edge(START, "chatting")
    graph.add_edge("chatting", END)

    # Compile workflow
    workflow = graph.compile()

    # Initial state
    initial_state = {"human_message": "", "AI_message": ""}

    # Invoke workflow
    final_state = workflow.invoke(initial_state)
    print("AI Response:", final_state['AI_message'])

# Call main dynamically
if __name__ == "__main__":
    main(query="Explain large language models in simple terms", temperature=0.9, max_output_tokens=400)
