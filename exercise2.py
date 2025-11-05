from typing import TypedDict, List
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    name : str
    value : list[int]
    operation : str
    result : int

def operator(state: AgentState) -> AgentState:
    if state["operation"] == "+":
        state["result"] = sum(state["value"])
    elif state["operation"] == "-":
        state["result"] = state["value"][0] - state["value"][1]
    elif state["operation"] == "*":
        state["result"] = state["value"][0] * state["value"][1]
    elif state["operation"] == "/":
        state["result"] = state["value"][0] / state["value"][1]
    return state




graph=StateGraph(AgentState)
graph.add_node("operator", operator)
graph.set_entry_point("operator")
graph.set_finish_point("operator")
app=graph.compile()
result=app.invoke({"name": "Mohammad", "value": [1, 2, 3], "operation": "-"})
print(f'Hi {result["name"]} ,  Your result is {result["result"]}')