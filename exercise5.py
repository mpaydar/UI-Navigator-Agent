from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
import random 


class AgentState(TypedDict):
    player_name: str
    guesses: list[int]
    attempts: int
    guess_limit: int
    target_number: int
    lower_bound: int
    upper_bound: int
   
def greet_player(state: AgentState) -> AgentState:
    state["player_name"] = input("Enter your name: ")
    state["target_number"] = random.randint(state["lower_bound"], state["upper_bound"])
    state["guesses"] = []
    state["attempts"] = 0
    print(f"Welcome to the game {state['player_name']}! Guess a number I have in mind which is between {state['lower_bound']} and {state['upper_bound']}.")
    return state

def guess_number(state: AgentState) -> AgentState:
    guess = int(input("Enter your guess: "))
    state["guesses"].append(guess)
    state["attempts"] += 1
    return state

def give_hint(state: AgentState) -> AgentState:
    if state["guesses"][-1] < state["target_number"]:
        print(f"Too low! Try again.")
        return state
    elif state["guesses"][-1] > state["target_number"]:
        print(f"Too high! Try again.")
        return state
    else:
        print(f"Congratulations {state['player_name']}! You guessed the number in {state['attempts']} attempts.")
        return state

def should_continue(state: AgentState) -> str:
    if state["guesses"][-1] == state["target_number"]:
        return "exit"
    elif state["attempts"] < state["guess_limit"]:
        return "loop"
    else:
        return "exit"

graph=StateGraph(AgentState)
graph.add_node("greet_player_node", greet_player)
graph.add_node("guess_number_node", guess_number)
graph.add_node("give_hint_node", give_hint)
graph.add_edge(START, "greet_player_node")
graph.add_edge("greet_player_node", "guess_number_node")
graph.add_edge("guess_number_node", "give_hint_node")
graph.add_conditional_edges("give_hint_node", should_continue, {
    "loop": "guess_number_node",
    "exit": END
})
graph.set_entry_point("greet_player_node")  
app=graph.compile()
result=app.invoke({"player_name": "Mohammad", "lower_bound": 1, "upper_bound": 20, "guess_limit": 7})
print(result)
