from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

class ChatAttribute(TypedDict):
    AI_message:str
    human_message:str


def chatting(state: ChatAttribute)-> ChatAttribute:
    query=state['human_message']


    llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    from langchain_core.output_parsers import StrOutputParser
    parser=StrOutputParser()

    chain=llm | parser

    response=chain.invoke(query)

    state['AI_message']=response

    return state

from langgraph.graph import StateGraph, START, END
graph=StateGraph(ChatAttribute)

graph.add_node("chatting", chatting)


graph.add_edge(START, "chatting")
graph.add_edge("chatting", END)

workflow=graph.compile()
initial_state={"human_message":"What is AI"}
final_state=workflow.invoke(initial_state)
print(final_state['AI_message'])
