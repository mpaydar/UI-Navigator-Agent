from typing import TypedDict, List
from langgraph.graph import StateGraph



class AgentState(TypedDict):
    name: str
    age: str 
    skills : list[str]
    result : str

def first_node(state: AgentState) -> AgentState:
    """This is the first node of our sequence"""
    state["result"] = f"Hi {state['name']}!"
    return state


def second_node(state: AgentState) -> AgentState:
    """This is the second node of our sequence"""
    state["result"] = state['result'] + f" You are {state['age']} years old!"
    return state

def third_node(state: AgentState) -> AgentState:
    """This is the third node of our sequence"""
    state["result"] = state["result"] + f"You have the following skills: {', '.join(state['skills'])}"
    return state

graph=StateGraph(AgentState)
graph.add_node("first_node", first_node)
graph.add_node("second_node", second_node)
graph.add_node("third_node", third_node)
graph.add_edge("first_node", "second_node")
graph.add_edge("second_node", "third_node")

graph.set_entry_point("first_node")
graph.set_finish_point("third_node")


app=graph.compile()
answer=app.invoke({"name": "Mohammad", "age": "20", "skills": ["Python", "Java", "C++"]})
print(answer["result"])