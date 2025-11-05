from langgraph.graph import StateGraph
from typing import Dict,TypedDict


class AgentState(TypedDict):
    message: str

def greeting_node(state : AgentState) -> AgentState:
    """Simple node that adds a greeting message to the state"""
    
    state["message"] = "Hey " + state["message"] + ", how is your day going?"
    return state



graph=StateGraph(AgentState)
graph.add_node("greeter", greeting_node)
graph.set_entry_point("greeter")
graph.set_finish_point("greeter")

app=graph.compile()
result=app.invoke({"message": "Mohammad"})
print(result["message"])