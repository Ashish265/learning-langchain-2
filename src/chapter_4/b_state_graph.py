from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]


builder = StateGraph(State)

model = ChatOpenAI()


def chatbot(state: State):
    answer = model.invoke(state['messages'])
    return {"messages": [answer]}


builder.add_node("chatbot", chatbot)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

# Run the graph
input = {"messages": [HumanMessage("hi")]}
for chunk in graph.stream(input):
    print(chunk)
