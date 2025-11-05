from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END


class AgentState(TypedDict):
    number1: int
    operation: str
    number2: int 
    finalNumber: int


def adder(state: AgentState) -> AgentState:
    state["finalNumber"] = state["number1"] + state["number2"]
    return state

def subtracter(state: AgentState) -> AgentState:
    state["finalNumber"] = state["number1"] - state["number2"]
    return state

def decide_next_node(state: AgentState) -> str:
    """Decide the next node based on the operation"""
    if state["operation"] == "+":
        return "addition_operation"
    else:
        return "subtraction_operation"

graph=StateGraph(AgentState)
graph.add_node("add_node", adder)
graph.add_node("subtract_node", subtracter)
graph.add_node("router", lambda state: state)  # passthrough function

graph.add_edge(START, "router")
graph.add_conditional_edges("router", decide_next_node ,
# edges 
{  "addition_operation": "add_node",
    "subtraction_operation": "subtract_node"
})
graph.add_edge("add_node", END)
graph.add_edge("subtract_node", END)

app=graph.compile()
initial_state=AgentState({"number1": 10, "operation": "-", "number2": 50})
print(app.invoke(initial_state))
# result=app.invoke({"number1": 10, "operation": "+", "number2": 50})
# print(result["finalNumber"])
