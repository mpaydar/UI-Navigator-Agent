from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END


class AgentState(TypedDict):
    number1: int
    operation: str
    number2: int 
    number3: int
    number4: int
    operation1: str
    operation2: str
    finalNumber: int


def adder(state: AgentState) -> AgentState:
    state["finalNumber"] = state["number1"] + state["number2"]
    return state

def adder2(state: AgentState) -> AgentState:
    state["finalNumber"] = state["number3"] + state["number4"]
    return state



def subtracter(state: AgentState) -> AgentState:
    state["finalNumber"] = state["number1"] - state["number2"]
    return state

def subtracter2(state: AgentState) -> AgentState:
    state["finalNumber"] = state["number3"] - state["number4"]
    return state

def decide_next_node(state: AgentState) -> str:
    """Decide the next node based on the operation"""
    if state["operation"] == "+":
        return "addition_operation"
    else:
        return "subtraction_operation"


def decide_next_node2(state: AgentState) -> str:
    """Decide the next node based on the operation"""
    if state["operation2"] == "+":
        return "addition_operation2"
    else:
        return "subtraction_operation2"



graph=StateGraph(AgentState)
graph.add_node("add_node", adder)
graph.add_node("subtract_node", subtracter)
graph.add_node("add_node2", adder2)
graph.add_node("subtract_node2", subtracter2)



graph.add_node("router", lambda state: state)
graph.add_node("router2", lambda state: state)
graph.add_edge(START, "router")
graph.add_conditional_edges("router", decide_next_node ,
# edges 
{  "addition_operation": "add_node",
    "subtraction_operation": "subtract_node"
})


graph.add_edge("add_node", "router2")
graph.add_edge("subtract_node", "router2")
graph.add_conditional_edges("router2", decide_next_node2 , {
    "addition_operation2": "add_node2",
    "subtraction_operation2": "subtract_node2"
})

graph.add_edge("add_node2", END)
graph.add_edge("subtract_node2", END)




app=graph.compile()
initial_state=AgentState({"number1": 10, "operation": "-", "number2": 50, "number3": 10, "number4": 20, "operation1": "+", "operation2": "-"})
print(app.invoke(initial_state))
# result=app.invoke({"number1": 10, "operation": "+", "number2": 50})
# print(result["finalNumber"])
