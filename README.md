# My proposal for the agent :

## Graph Representation
<img width="1016" height="1292" alt="image" src="https://github.com/user-attachments/assets/3ea898bc-62e9-4a92-b45c-2ff9b721cc8c" />

## agent2 is the main file to run it make sure you type python agent2.py in the terminal . You might need to install the dependencies!

A. The Inspector will be PlayWright

B. The Actor will be PlayWright

C. The Planner will be openAI 

### Human Message: can be any task to be perform on any platform

     Example: Show me how to create a project on linear platform

### System Message:  You are a smart planner. You ask for the current UI content from the inspector. Then, You choose the appropriate selector according to the current goal and you pass the selector to the actor to take action. You will be recursively perform the following steps: 

  1. Ask the current content of the UI
  2. Select the appropariate selector
  3. Send the selector to the actor to get coser to the goal

