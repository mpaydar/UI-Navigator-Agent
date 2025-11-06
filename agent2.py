from typing import TypedDict, Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, Page
from openai import OpenAI
import time
import base64
import json
import re
load_dotenv()
i=0
# Global playwright resources (will be initialized in main)
_playwright = None
_browser = None
_context = None
_page = None

def get_page() -> Page:
    """Get or create page instance"""
    global _playwright, _browser, _context, _page
    if _page is None:
        _playwright = sync_playwright().start()
        _browser = _playwright.chromium.launch(headless=False)
        _context = _browser.new_context(viewport={"width": 1280, "height": 800})
        _page = _context.new_page()
    return _page

def cleanup_browser():
    """Clean up playwright resources"""
    global _playwright, _browser, _context, _page
    if _context:
        _context.close()
    if _browser:
        _browser.close()
    if _playwright:
        _playwright.stop()
    _page = None
    _context = None
    _browser = None
    _playwright = None

class AgentState(TypedDict):
    messages: list[Union[HumanMessage, SystemMessage, AIMessage]]
    screenshot: str
    img_base64: str
    goal: str
    selector: str
    action_type: str  # 'click' or 'type'
    action_text: str  # Text to type (if action_type is 'type')
    visible_buttons: list[str]
    visible_inputs: list[str]
    is_first_visit: bool  # Track if this is the first inspection

def inspector(state: AgentState) -> AgentState:
    """Navigate to website and take screenshot"""
    page = get_page()
    is_first = state.get("is_first_visit", True)
    
    if is_first:
        print("📸 Inspector: Navigating to Linear...")
        page.goto("https://www.notion.com/", wait_until="load", timeout=60000)
        page.wait_for_timeout(2000)  # Let page hydrate
        state["is_first_visit"] = False
    else:
        print("📸 Inspector: Taking screenshot of current page...")
        page.wait_for_timeout(1000)  # Wait for animations to settle
        
    # Check for authentication pages - pause for manual login
    current_url = page.url
    # Use regex to detect login/auth pages (case-insensitive)
    login_pattern = re.compile(r'(login|signin|sign-in|signup|sign-up|auth|authenticate|register)', re.IGNORECASE)
    if login_pattern.search(current_url):
        print("\n" + "="*70)
        print("🔐 AUTHENTICATION REQUIRED")
        print("="*70)
        print(f"Current URL: {current_url}")
        print("\nPlease log in MANUALLY in the browser window.")
        print("="*70)
        input("\nPress ENTER after you've logged in: ")
        page.wait_for_timeout(2000)
        print(f"\n✅ Continuing from: {page.url}\n")
    
    # Get all visible buttons and links
    visible_buttons = page.locator("button:visible, a:visible").all()
    state["visible_buttons"] = [el.inner_text().strip() for el in visible_buttons if el.inner_text().strip()]
    
    print("\nVISIBLE BUTTONS / LINKS:")
    for i, text in enumerate(state["visible_buttons"][:10]):
        print(f"{i+1}. {text}")

    # Get visible inputs INCLUDING contenteditable divs (rich text editors)
    visible_inputs = page.locator("input:visible, textarea:visible, [contenteditable='true']:visible, [role='textbox']:visible").all()
    input_descriptors = []
    
    print("\nVISIBLE INPUT FIELDS:")
    for i, inp in enumerate(visible_inputs[:15]):
        try:
            tag_name = inp.evaluate("el => el.tagName.toLowerCase()")
            inp_type = inp.get_attribute("type") or "text"
            name = inp.get_attribute("name") or ""
            placeholder = inp.get_attribute("placeholder") or ""
            aria_label = inp.get_attribute("aria-label") or ""
            role = inp.get_attribute("role") or ""
            
            # Create detailed descriptor
            if tag_name == "div" and role == "textbox":
                descriptor = f"Editable div (rich text)"
                if aria_label:
                    descriptor += f" aria-label='{aria_label}'"
                    print(f"{i+1}. Rich text field: aria-label='{aria_label}'")
                else:
                    print(f"{i+1}. Rich text field (contenteditable div)")
            else:
                descriptor = f"Input type={inp_type}"
                if placeholder:
                    descriptor += f" placeholder='{placeholder}'"
                    print(f"{i+1}. {tag_name}: placeholder='{placeholder}', Name: {name}")
                elif name:
                    descriptor += f" name='{name}'"
                    print(f"{i+1}. {tag_name}: name='{name}'")
                elif aria_label:
                    descriptor += f" aria-label='{aria_label}'"
                    print(f"{i+1}. {tag_name}: aria-label='{aria_label}'")
                else:
                    print(f"{i+1}. {tag_name} type={inp_type}")
            
            input_descriptors.append(descriptor)
        except:
            pass
    
    state["visible_inputs"] = input_descriptors
    
    # Take screenshot
    page.screenshot(path="step_current.png")
    with open("step_current.png", "rb") as f:
        state["img_base64"] = base64.b64encode(f.read()).decode("utf-8")
    print("📸 Screenshot saved: step_current.png\n")
    
    return state

       
def planner(state: AgentState) -> AgentState:
    """Analyze screenshot with GPT-4 Vision and plan next action"""
    print("🤖 Planner: Analyzing screenshot with GPT-4 Vision...")
    all_visible_elements = state["visible_buttons"] + state["visible_inputs"]
    all_visible_elements_string = "\n".join(all_visible_elements)

    client = OpenAI()
    
    elements_context = f"\nVISIBLE ELEMENTS ON PAGE:\n{all_visible_elements_string}" if all_visible_elements else "\nNo elements extracted."
    
    system_message = (
        "You are an automation planner analyzing screenshots and DOM elements. "
        "Return JSON with 'selector' key. "
        "CRITICAL: Goal is complete ONLY when the full task is done (logged in AND at dashboard, project created AND saved). "
        "Seeing a button to click is NOT completion - you must click it and complete the task. "
        f"{elements_context}"
    )
    
    # Send image to GPT-4o for visual understanding
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Analyze this screenshot of the Linear app.

User Goal: {state["goal"]}

Instructions:
1. Analyze the screenshot - what do you see?
2. Is the goal "{state["goal"]}" FULLY complete? (NOT just "button visible", but task DONE)
3. If NO, what is the NEXT action to progress?

Return JSON with these keys:
- "action_type": "click" (for buttons/links) or "type" (for input fields)
- "selector": Playwright selector for the element
- "action_text": Text to type (if action_type is "type"), otherwise empty string

CRITICAL:
- Goal complete ONLY when task is DONE (logged in AND at dashboard, project created AND saved)
- If goal complete: {{"action_type": "click", "selector": "", "action_text": ""}}
- If need to click: {{"action_type": "click", "selector": "text=Log in", "action_text": ""}}
- If need to type into rich text field: {{"action_type": "type", "selector": "[aria-label='Project name']", "action_text": "bayat123456"}}
- If need to type into regular input: {{"action_type": "type", "selector": "input[placeholder='Add a short summary']", "action_text": "Description here"}}
- For project name from goal "{state["goal"]}", extract the name (e.g., "bayat123456") and use it
- SELECTOR RULES:
  * For "Rich text field: aria-label='X'" use: [aria-label='X'] or [role='textbox'][aria-label='X']
  * For "Input placeholder='X'" use: input[placeholder='X'] or textarea[placeholder='X']
  * For buttons/links: text=, button:has-text(), [data-testid='...']
- ONLY use elements from the VISIBLE ELEMENTS list above
- If element not found, modal may be closed - suggest opening it first

Return valid JSON only."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{state['img_base64']}"
                        }
                    }
                ]
            }
        ],
        max_tokens=500
    )
    
    # Parse response
    raw_content = response.choices[0].message.content
    
    if not raw_content:
        print("⚠️  GPT-4 Vision returned empty response")
        state["selector"] = ""
        return state
    
    try:
        analysis_json = json.loads(raw_content)
        action_type = analysis_json.get("action_type", "click").lower()
        selector = analysis_json.get("selector", "")
        action_text = analysis_json.get("action_text", "")
        
        print("\n" + "="*70)
        print("🤖 GPT-4 Vision Decision:")
        print("="*70)
        print(f"Action Type: {action_type}")
        print(f"Selector: {selector}")
        if action_type == "type":
            print(f"Text to Type: {action_text}")
        print("="*70 + "\n")
        
        state["action_type"] = action_type
        state["selector"] = selector
        state["action_text"] = action_text
        
    except json.JSONDecodeError as e:
        print(f"⚠️  Failed to parse JSON: {e}")
        print(f"Raw response: {raw_content}")
        # Fallback: assume click action
        state["action_type"] = "click"
        state["selector"] = raw_content.strip() if raw_content else ""
        state["action_text"] = ""
    
    return state

def executor(state: AgentState) -> AgentState:
    """Execute the action - click or type based on action_type"""
    global i
    print("🤖 Executor: Executing the action...")
    page = get_page()
    i+=1
    page.screenshot(path=f"screenshots/step_{i}.png")
    selector = state.get("selector", "").strip()
    action_type = state.get("action_type", "click").lower()
    action_text = state.get("action_text", "")
    
    if not selector:
        print("⚠️  No selector provided, skipping action")
        return state
    
    try:
        # Wait a bit for any modals/animations to settle
        page.wait_for_timeout(800)
        
        # Try to locate the element with wait
        print(f"🔍 Looking for element: {selector}")
        
        # Wait for element to be attached to DOM (give modals time to appear)
        try:
            page.wait_for_selector(selector, state="attached", timeout=5000)
        except Exception as wait_err:
            print(f"⚠️  Element not found after waiting, trying anyway: {wait_err}")
        
        loc = page.locator(selector)
        
        # Handle multiple matches - use first visible element
        count = loc.count()
        if count > 1:
            print(f"⚠️  Found {count} matching elements, using first one")
            loc = loc.first
        elif count == 0:
            # Try with case-insensitive search for placeholders
            if "placeholder=" in selector.lower():
                print("⚠️  Exact placeholder not found, trying case-insensitive...")
                # Extract placeholder text
                match = re.search(r"placeholder=['\"](.+?)['\"]", selector, re.IGNORECASE)
                if match:
                    placeholder_text = match.group(1)
                    # Try to find any input with similar placeholder
                    all_inputs = page.locator("input, textarea").all()
                    for inp in all_inputs:
                        inp_placeholder = inp.get_attribute("placeholder") or ""
                        if placeholder_text.lower() in inp_placeholder.lower():
                            print(f"✓ Found input with placeholder: {inp_placeholder}")
                            loc = page.locator(f"input[placeholder*='{placeholder_text}' i], textarea[placeholder*='{placeholder_text}' i]").first
                            break
                    else:
                        raise Exception(f"No input with placeholder containing '{placeholder_text}'")
                else:
                    raise Exception(f"No elements found for selector: {selector}")
            else:
                raise Exception(f"No elements found for selector: {selector}")
        
        if action_type == "type":
            # Type action: fill input field
            print(f"📝 Typing into: {selector}")
            print(f"Text: '{action_text}'")
            
            # Wait for element to be visible and interactable
            loc.wait_for(state="visible", timeout=5000)
            
            # Check if it's a contenteditable div (rich text editor)
            is_contenteditable = loc.evaluate("el => el.contentEditable === 'true'")
            
            if is_contenteditable:
                print("  (contenteditable div detected - using keyboard input)")
                # Focus the element
                loc.click(timeout=3000)
                # Select all existing text and delete
                page.keyboard.press("Meta+A")  # Cmd+A on Mac
                page.keyboard.press("Backspace")
                # Type the new text
                page.keyboard.type(action_text, delay=50)
            else:
                # Regular input/textarea
                print("  (regular input field)")
                loc.click(timeout=3000)
                loc.fill(action_text)  # fill() is faster for regular inputs
            
            page.wait_for_timeout(500)
            print(f"✓ Typed '{action_text}' successfully")
            
        else:
            # Click action (default)
            print(f"🖱️  Clicking: {selector}")
            
            # Wait for element to be visible and clickable
            loc.wait_for(state="visible", timeout=5000)
            
            loc.click(timeout=5000)
            page.wait_for_timeout(1500)  # Wait for navigation/modal
            print(f"✓ Click successful! Current URL: {page.url}")
            
    except Exception as e:
        error_msg = str(e)[:300]
        print(f"❌ Action failed: {error_msg}")
    
    return state

def set_goal(state: AgentState) -> AgentState:
    """Set the goal of the agent"""
    state["goal"] = input("Enter the goal of the agent: ")
    return state
def decide_next_action(state: AgentState) -> str:
    """Decide the next action based on the state"""
    if state["selector"] == "":
        return "end"
    else:
        return "next_action"


# Build graph
graph = StateGraph(AgentState)
graph.add_node("goal", set_goal)
graph.add_node("inspector", inspector)
graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_edge(START, "goal")
graph.add_edge("goal", "inspector")
graph.add_edge("inspector", "planner")
graph.add_edge("planner", "executor")
graph.add_conditional_edges("executor", decide_next_action, {
    "next_action": "inspector",
    "end": END
})

app = graph.compile()

if __name__ == "__main__":
    try:
        init_state = {
            "messages": [HumanMessage(content="I want to create a new project")],
            "screenshot": "",
            "img_base64": "",
            "goal": "",
            "selector": "",
            "action_type": "click",
            "action_text": "",
            "visible_buttons": [],
            "visible_inputs": [],
            "is_first_visit": True
        }
        
        config = {"recursion_limit": 50}
        final_state = app.invoke(init_state, config)
        
        print("\n" + "="*70)
        print("COMPLETE")
        print("="*70)
        print(f"Goal: {final_state.get('goal')}")
        print(f"Messages exchanged: {len(final_state['messages'])}")
        
    finally:
        # Clean up Playwright resources
        print("\n🧹 Cleaning up...")
        cleanup_browser()
