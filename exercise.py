from langgraph.graph import StateGraph
from typing import Dict,TypedDict

class AgentState(TypedDict):
    name: str

def complement(state:AgentState) -> AgentState:

    state["name"] = state["name"] + " You are doing  an amazing job learning  LangGraph"
    return state

graph=StateGraph(AgentState)
graph.add_node("complementer", complement)
graph.set_entry_point("complementer")
graph.set_finish_point("complementer")

app=graph.compile()
result=app.invoke({"name": "Mohammad"})
print(result["name"])