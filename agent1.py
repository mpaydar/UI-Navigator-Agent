from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv


load_dotenv()
class AgentState(TypedDict):
    messages: list[HumanMessage]

llm = ChatOpenAI(model="gpt-4o")
def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\n\nResponse: {response.content}")
    return state

graph=StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

app=graph.compile()

user_input=input("Enter your message: ")
while user_input.lower() != "exit":
    app.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input=input("Enter your message: ")

