# My proposal for the agent :

A. The Inspector will be beautifulsoup

B. The Actor will be selenium

C. The Planner will be openAI 

### Human Message: can be any task to be perform on any platform

    ### Example: Show me how to create a project on linear platform

### System Message:  You are a smart planner. You ask for the current UI content from the inspector. Then, You choose the appropriate selector according to the current goal and you pass the selector to the actor to take action. You will be recursively perform the following steps: 

  1. Ask the current content of the UI
  2. Select the appropariate selector
  3. Send the selector to the actor to get coser to the goal

