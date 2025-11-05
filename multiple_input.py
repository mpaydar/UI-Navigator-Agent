from typing import TypedDict, List
from langgraph.graph import StateGraph



class AgentState(TypedDict):
    values: List[int]
    name: str
    result: int


def  process_values(state: AgentState) -> AgentState:
   
    """Process the values and return the result"""
    state["result"] = f"Hi there {state['name']} ! Your Sum =  {sum(state['values'])}"
    return state


graph=StateGraph(AgentState)
graph.add_node("process_values", process_values)
graph.set_entry_point("process_values")
graph.set_finish_point("process_values")
app=graph.compile()

result=app.invoke({"values": [1, 2, 3], "name": "Mohammad "})
print(result["result"])