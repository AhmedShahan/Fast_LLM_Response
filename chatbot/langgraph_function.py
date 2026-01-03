from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from functools import partial

class ChatAttribute(TypedDict):
    AI_message: str
    human_message: str
    
def langGraphChatbot(temperature=0.7, max_output_tokens=500):
    from functools import partial
    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.output_parsers import StrOutputParser

    class ChatAttribute(TypedDict):
        AI_message: str
        human_message: str

    def chatting(state: ChatAttribute, temperature, max_output_tokens) -> ChatAttribute:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        query = state['human_message']
        parser = StrOutputParser()
        chain = llm | parser

        response = chain.invoke(query)
        state['AI_message'] = response
        return state

    graph = StateGraph(ChatAttribute)
    node_function = partial(chatting, temperature=temperature, max_output_tokens=max_output_tokens)
    graph.add_node("chatting", node_function)
    graph.add_edge(START, "chatting")
    graph.add_edge("chatting", END)

    workflow = graph.compile()
    initial_state = {"human_message": "What is AI", "AI_message": ""}
    final_state = workflow.invoke(initial_state)
    print(final_state['AI_message'])
if __name__ == "__main__":
    langGraphChatbot(temperature=0.9, max_output_tokens=400)
