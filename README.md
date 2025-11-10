<img width="1600" height="800" alt="image" src="https://github.com/user-attachments/assets/53b4be81-5ca0-4ee9-b84e-c1a56520910c" />


## What the Agent does?

It will let you to write any task on any platform in English and will execute it directly live on the UI. The UI will be provide you screen shot of every steps as the prove of its execution. 


## Current Capabilities

### Goal Interpretation:

Understands user intents like “Change the start day of the week to Monday in Notion.”

Autonomous Execution: Uses Playwright for live browser interaction, clicking, typing, and navigating across pages.

Visual Feedback: Captures step-by-step screenshots for traceability and reasoning visualization.

LangGraph Integration: Manages reasoning loops, planning, and context updates through structured state graphs.

Reasoning Pipeline: Uses GPT-4 Vision to analyze visual states and refine next-step decisions based on UI content.


## Future Capabilities: 

### Phase 1: Architectural Refinement

Implement the GAME model (Goals, Actions, Memory, Environment) to standardize internal agent components.

Design a modular Agent Core in Python, allowing future expansion (e.g.,multiple agents collaborating).

Add a persistent state layer to track environment context, previous errors, and success history.


### Phase 2: Memory and Context Management

Integrate vector-based memory (e.g., Chroma or FAISS) to store prior DOM states, action results, and failure patterns.

Implement context recall, enabling the agent to compare current tasks against past experiences to avoid repeated mistakes.


### Phase 3: Precision and Learning

Introduce self-reflection (Reflexion pattern) nodes: after each action, the agent critiques its performance and proposes corrective steps.

Apply reinforcement learning principles to build a lightweight reward model for evaluating successful outcomes.

Use Vanderbilt’s lessons on tool calling and environment interaction to improve policy refinement and adaptive control.



### Phase 4: Prompt Optimization & Evaluation

Employ DSPy and LangFuse to optimize prompts dynamically based on feedback and trace analytics.

Develop a structured evaluation suite that measures precision, goal completion rate, and action success over multiple test websites.



### Phase 5: Deployment & Visualization

Deploy a hosted demo version showing real-time navigation with transparent reasoning steps and accuracy tracking.









## Graph Representation
<img width="1016" height="1292" alt="image" src="https://github.com/user-attachments/assets/3ea898bc-62e9-4a92-b45c-2ff9b721cc8c" />


A. The Inspector will be PlayWright

B. The Actor will be PlayWright

C. The Planner will be openAI 

### Human Message: can be any task to be perform on any platform

     Example: Show me how to create a project on linear platform

### System Message:  You are a smart planner. You ask for the current UI content from the inspector. Then, You choose the appropriate selector according to the current goal and you pass the selector to the actor to take action. You will be recursively perform the following steps: 

  1. Ask the current content of the UI
  2. Select the appropariate selector
  3. Send the selector to the actor to get coser to the goal

