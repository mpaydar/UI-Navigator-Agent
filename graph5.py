from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
import random 




class AgentState(TypedDict):
    name: str
    number: list[int]
    counter: int


def greeting_node (state: AgentState) -> AgentState:
    state["name"] = f'Hi there, {state["name"]}'
    state["counter"] = 0
    return state



def random_node(state: AgentState) -> AgentState:
    """Generate a random number between 0 and 10"""
    state["number"] = random.randint(0, 10)
    state["counter"] += 1
    return state

def should_continue(state: AgentState) -> str:
    """function to decide what to do next"""
    if state["counter"] < 5:
        print(f"Entering Loop, {state['counter']}")
        return "loop"
    else:
        return "exit"

graph=StateGraph(AgentState)
graph.add_node("greeting_node", greeting_node)
graph.add_node("random_node", random_node)
graph.add_edge("greeting_node", "random_node")

graph.add_conditional_edges("random_node", should_continue, 
{
    # edges 
    "loop": "random_node",
    "exit": END
})

graph.set_entry_point("greeting_node")
app=graph.compile()
result=app.invoke({"name": "Mohammad"})
print(result)