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
import os
load_dotenv()

# Create screenshots directory
from pathlib import Path
os.makedirs("screenshots", exist_ok=True)

# Global variables
i = 0
screenshots_dir = Path("screenshots")  # Will be updated per task in set_goal()

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

def ask_gpt_for_better_regex(goal: str, failed_pattern: str, img_base64: str, visible_elements: list) -> str:
    """Ask GPT Vision to suggest a better regex pattern by analyzing the screenshot"""
    print("🔍 Asking GPT Vision for a better regex pattern...")
    
    elements_list = "\n".join([f"{i+1}. [{el['role']}] {el['name']}" for i, el in enumerate(visible_elements[:30])])
    
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""GOAL: {goal}

The pattern "{failed_pattern}" didn't find any element.

Look at this screenshot and the list of elements below.
Suggest a BETTER regex pattern that will match an element to accomplish the goal.

ELEMENTS:
{elements_list}

Return ONLY the regex pattern (no quotes, no explanations).
Examples: "appearance", "theme|appearance", "dark\\s*mode"

NO inline flags like (?i) - not supported!"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=50
    )
    
    suggested_pattern = response.choices[0].message.content.strip()
    
    # Strip markdown code blocks if present
    if suggested_pattern.startswith("```") and suggested_pattern.endswith("```"):
        lines = suggested_pattern.split("\n")
        if len(lines) >= 3:  # Has opening, content, and closing
            suggested_pattern = "\n".join(lines[1:-1]).strip()
    
    # Strip quotes
    suggested_pattern = suggested_pattern.strip('"').strip("'")
    
    print(f"💡 GPT Vision suggests: '{suggested_pattern}'")
    return suggested_pattern

def check_toggle_state_from_screenshot(goal: str, element_name: str, img_base64: str) -> tuple[bool, str]:
    """
    Use GPT-4 Vision to analyze screenshot and determine toggle state.
    Returns (is_goal_achieved, visual_state_description)
    FULLY GENERAL - uses JSON response, no hardcoded phrase matching!
    """
    client = OpenAI()
    
    prompt = f"""Look at this screenshot and analyze the element: "{element_name}"

User's goal: "{goal}"

Return a JSON object:
{{
  "current_visual_state": "describe what you see",
  "goal_satisfied": true/false
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                        }
                    ]
                }
            ],
            max_tokens=150,
            temperature=0
        )
        
        result_json = json.loads(response.choices[0].message.content.strip())
        
        visual_state = result_json.get("current_visual_state", "Unknown")
        is_achieved = result_json.get("goal_satisfied", False)
        
        print(f"👁️  GPT Vision: {visual_state}")
        print(f"👁️  GPT Vision: Goal satisfied = {is_achieved}")
        
        return is_achieved, visual_state
    except Exception as e:
        print(f"⚠️  Vision check failed: {e}")
        return False, "Unknown (vision check failed)"

def check_goal_achieved_by_state(goal: str, element_name: str, current_state: str) -> bool:
    """
    Use GPT-4o-mini to determine if current element state satisfies the goal.
    Returns True if goal is achieved, False otherwise.
    FULLY GENERAL - uses JSON response, no hardcoded phrase matching!
    """
    client = OpenAI()
    
    # Don't trust this check if state is unclear
    if "unknown" in current_state.lower() or "unclear" in current_state.lower() or "infer" in current_state.lower():
        print(f"⚠️  State is unclear - skipping text-based check, will use vision")
        return False
    
    prompt = f"""User's goal: "{goal}"

Element: "{element_name}"
Current state: {current_state}

Analyze whether this current state satisfies the user's goal.

Return JSON:
{{
  "goal_satisfied": true/false,
  "reasoning": "brief explanation"
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0
        )
        
        result_json = json.loads(response.choices[0].message.content.strip())
        
        is_achieved = result_json.get("goal_satisfied", False)
        reasoning = result_json.get("reasoning", "No reasoning provided")
        
        print(f"🧠 GPT: {reasoning}")
        print(f"🧠 GPT: Goal satisfied = {is_achieved}")
        
        return is_achieved
    except Exception as e:
        print(f"⚠️  Goal check failed: {e}")
        return False

def find_semantic_match(goal: str, visible_elements: list[dict]) -> dict:
    """
    Use GPT-4o-mini to semantically match the goal against visible elements.
    Returns the best matching element or None.
    FULLY GENERAL - uses JSON response!
    """
    client = OpenAI()
    
    # Build a simple list of visible elements
    elements_list = []
    for i, el in enumerate(visible_elements[:30]):
        elements_list.append(f"{i+1}. [{el['role']}] {el['name']}")
    
    elements_str = "\n".join(elements_list) if elements_list else "No elements available"
    
    prompt = f"""User goal: "{goal}"

Visible UI elements:
{elements_str}

Which element best helps achieve the goal?

Return JSON:
{{
  "element_number": 5,
  "reasoning": "brief explanation"
}}

If no good match, use element_number: 0"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0
        )
        
        result_json = json.loads(response.choices[0].message.content.strip())
        
        element_num = result_json.get("element_number", 0)
        reasoning = result_json.get("reasoning", "")
        
        if element_num == 0 or element_num > len(visible_elements):
            print(f"🧠 GPT: No suitable element found - {reasoning}")
            return None
        
        idx = element_num - 1  # Convert to 0-based index
        matched_el = visible_elements[idx]
        print(f"🧠 GPT Semantic Match: [{matched_el['role']}] {matched_el['name']}")
        print(f"🧠 Reasoning: {reasoning}")
        return matched_el
    except Exception as e:
        print(f"⚠️  Semantic matching failed: {e}")
        return None

class AgentState(TypedDict):
    messages: list[Union[HumanMessage, SystemMessage, AIMessage]]
    screenshot: str
    img_base64: str
    goal: str
    website_url: str  # Starting URL (platform-agnostic)
    role: str  # Accessibility role: 'link', 'button', 'textbox', etc.
    name_pattern: str  # Regex pattern for the accessible name
    action_type: str  # 'click' or 'type' or 'keyboard' or 'hover' or 'noop'
    action_text: str  # Text to type (if action_type is 'type') or key to press (if 'keyboard')
    visible_elements: list[dict]  # List of {role, name, description}
    is_first_visit: bool  # Track if this is the first inspection
    actions_performed: list[str]  # Track all successful actions to avoid loops
    failed_actions: list[str]  # Track all failed actions
    goal_text_entered: bool  # Flag when goal is complete
    last_url: str  # Track URL changes to detect when stuck
    hover_explored: list[str]  # Track which elements we've hovered over for exploration

def inspector(state: AgentState) -> AgentState:
    """Navigate to website and take screenshot"""
    page = get_page()
    is_first = state.get("is_first_visit", True)
    website_url = state.get("website_url", "")
    
    if is_first:
        if not website_url:
            print("❌ No website URL provided!")
            return state
        
        print(f"📸 Inspector: Navigating to {website_url}...")
        page.goto(website_url, wait_until="load", timeout=60000)
        page.wait_for_timeout(2000)  # Let page hydrate
        state["is_first_visit"] = False
        print(f"✅ Loaded: {page.url}")
    else:
        print(f"📸 Inspector: Current page - {page.url}")
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
    
    # Extract ALL interactive elements using accessibility tree
    visible_elements = []
    print("\nVISIBLE INTERACTIVE ELEMENTS:")
    
    # Check for modals/dialogs first (they have priority)
    modals = page.locator("[role='dialog']:visible, [role='alertdialog']:visible").all()
    
    if modals:
        print(f"  ℹ️  {len(modals)} modal(s) detected - PRIORITIZING MODAL CONTENT")
    
    # Use accessibility tree to get ALL interactive elements (most reliable!)
    try:
        tree = page.accessibility.snapshot()
        
        def extract_elements(node, depth=0):
            """Recursively extract clickable elements from accessibility tree"""
            if not isinstance(node, dict):
                return
            
            role = node.get("role", "")
            name = node.get("name", "")
            disabled = node.get("disabled", False)
            
            # Check if element is interactive
            interactive_roles = [
                "link", "button", "textbox", "switch", "tab", 
                "option", "menuitem", "menuitemradio", "menuitemcheckbox",
                "treeitem", "combobox", "listbox", "row", "cell", "gridcell"
            ]
            
            # Skip disabled elements (like table headers)
            if disabled:
                return
            
            if role in interactive_roles and name and len(name) < 100:
                # Skip table column headers (they're usually buttons but not useful)
                skip_patterns = ["hidden properties", "hidden columns", "drag"]
                if any(skip in name.lower() for skip in skip_patterns):
                    return
                
                # Skip if already in list (avoid duplicates)
                if not any(el['name'] == name and el['role'] == role for el in visible_elements):
                    # Highlight "New" or "Add" buttons for tasks/todos
                    if role == "button" and ("new" in name.lower() and ("task" in name.lower() or "to-do" in name.lower() or "todo" in name.lower())):
                        name = f"🆕 {name}"  # Mark as important
                    
                    # Highlight user profile/account menuitems (these often lead to settings)
                    if role == "menuitem" and ("'s" in name or "profile" in name.lower() or "account" in name.lower()):
                        name = f"📋{name}"  # Mark as profile menu
                    
                    visible_elements.append({
                        "role": role,
                        "name": name,
                        "description": f"{role} '{name}'"
                    })
            
            # Recursively process children
            for child in node.get("children", []):
                if isinstance(child, dict):
                    extract_elements(child, depth + 1)
        
        extract_elements(tree)
        
        # Supplement with direct locator search for buttons missed by accessibility tree
        try:
            # Search for all buttons containing task-related text
            all_buttons = page.locator("button:visible, div[role='button']:visible, a:visible").all()
            for btn in all_buttons[:50]:
                try:
                    text = btn.inner_text().strip()
                    aria_label = btn.get_attribute("aria-label") or ""
                    display_name = text or aria_label
                    
                    if display_name and len(display_name) < 80:
                        # Look for task/todo creation buttons
                        is_task_button = (
                            ("new" in display_name.lower() and "task" in display_name.lower()) or
                            ("add" in display_name.lower() and "task" in display_name.lower()) or
                            ("+" in display_name and "task" in display_name.lower()) or
                            (display_name.lower() == "+ new task") or
                            ("new to-do" in display_name.lower())
                        )
                        
                        # Also look for general "add" buttons when not already captured
                        is_general_add = "+" in display_name and len(display_name) < 15
                        
                        if (is_task_button or is_general_add) and not any(el['name'] == display_name or el['name'] == f"🆕 {display_name}" for el in visible_elements):
                            visible_elements.append({
                                "role": "button",
                                "name": f"🆕 {display_name}",
                                "description": f"button '{display_name}'"
                            })
                            print(f"  💡 Found via locator: {display_name}")
                except:
                    pass
        except Exception as e:
            print(f"⚠️ Supplementary search failed: {e}")
        
        # Print elements
        for i, el in enumerate(visible_elements[:40]):
            role = el['role']
            name = el['name']
            
            # Highlight important elements
            if role == "switch":
                print(f"{i+1}. [🔘{role}] {name}")
            elif role in ["option", "menuitem", "menuitemradio"]:
                print(f"{i+1}. [📋{role}] {name}")
            else:
                print(f"{i+1}. [{role}] {name}")
                
    except Exception as e:
        print(f"⚠️ Error extracting from accessibility tree: {e}")
    
    state["visible_elements"] = visible_elements
    
    print(f"\n📊 Total elements extracted: {len(visible_elements)}")
    print(f"   Roles breakdown: {', '.join(set(el['role'] for el in visible_elements))}")
    
    # Debug: Check if we're in a table view (lots of column buttons)
    column_buttons = [el for el in visible_elements if el['role'] == 'button' and any(col in el['name'].lower() for col in ['task name', 'status', 'assignee', 'due date', 'priority'])]
    if len(column_buttons) >= 3:
        print(f"\n📋 Table detected with {len(column_buttons)} column headers")
        print("   💡 Hint: Look for 'New' or 'Add' button to create a row, or find a textbox to type directly")
    
    # Debug: Check if theme-related elements are captured
    theme_elements = [el for el in visible_elements if "dark" in el['name'].lower() or "light" in el['name'].lower() or "theme" in el['name'].lower()]
    if theme_elements:
        print(f"\n🎨 Theme-related elements found:")
        for el in theme_elements:
            print(f"   [{el['role']}] {el['name']}")
    
    # Take screenshot
    global screenshots_dir
    current_screenshot = screenshots_dir / "step_current.png"
    page.screenshot(path=str(current_screenshot))
    with open(current_screenshot, "rb") as f:
        state["img_base64"] = base64.b64encode(f.read()).decode("utf-8")
    print(f"📸 Screenshot saved: {current_screenshot}\n")
    
    return state

       
def planner(state: AgentState) -> AgentState:
    """Analyze screenshot with GPT-4 Vision and plan next action"""
    print("🤖 Planner: Analyzing screenshot with GPT-4 Vision...")
    page = get_page()
    
    # OPTIMIZATION 4: Early termination if goal complete
    if state.get("goal_text_entered", False):
        print("✅ Goal already complete! Terminating.")
        state["role"] = ""
        state["name_pattern"] = ""
        return state
    
    visible_elements = state.get("visible_elements", [])
    img_base64 = state.get("img_base64", "")
    
    # Ensure we have a screenshot
    if not img_base64:
        print("❌ No screenshot available!")
        state["role"] = ""
        state["name_pattern"] = ""
        return state
    
    # OPTIMIZATION 5: Get action history
    failed_actions = state.get("failed_actions", [])
    actions_performed = state.get("actions_performed", [])

    client = OpenAI()
    
    # Build elements list for GPT
    elements_list = []
    for i, el in enumerate(visible_elements[:25]):
        elements_list.append(f"{i+1}. [{el['role']}] {el['name']}")
    
    elements_context = "\n".join(elements_list) if elements_list else "No elements found."
    failed_context = "\n⚠️ FAILED:\n" + "\n".join(failed_actions[-10:]) if failed_actions else ""
    actions_context = "\n✓ DONE:\n" + "\n".join(actions_performed[-10:]) if actions_performed else ""
    
    # Check if we just did a recovery (pressed Escape)
    just_recovered = any("recovery:escape" in act for act in actions_performed[-2:]) if actions_performed else False
    if just_recovered:
        actions_context += "\n\n⚠️ JUST RECOVERED: Pressed Escape to close wrong modal/menu. Try HOVER to explore or click a DIFFERENT element!"
    
    system_message = f"""You are a UI element identifier for accessibility purposes.

Output ONE JSON object:
{{
  "action_type": "click" | "type" | "keyboard" | "hover" | "noop",
  "role": "exact role from list",
  "name_pattern": "regex to match element name",
  "action_text": "text to type OR key name (Enter, Tab, Escape)"
}}

IMPORTANT: Think semantically! Consider synonyms and platform-specific naming:
- "dark mode" = "dark theme" = "black theme" = "appearance settings"
- "add task" = "new task" = "create item" = "+ Add" = "New"
- "settings" = "preferences" = "options" = "account settings"

Pattern syntax (case-insensitive by default):
- "log\\s*in" matches "Log in"
- "dark|appearance" matches either "dark" or "appearance"
- "new|add|create" matches any creation button

Examples:
{{"action_type":"click","role":"link","name_pattern":"log\\\\s*in","action_text":""}}
{{"action_type":"click","role":"button","name_pattern":"settings","action_text":""}}
{{"action_type":"click","role":"menuitem","name_pattern":".*'s$","action_text":""}}
{{"action_type":"click","role":"button","name_pattern":"empty\\\\s*page","action_text":""}}
{{"action_type":"click","role":"menuitem","name_pattern":"dark","action_text":""}}
{{"action_type":"hover","role":"button","name_pattern":"more.*options","action_text":""}}
{{"action_type":"type","role":"textbox","name_pattern":".*","action_text":"My text"}}
{{"action_type":"type","role":"gridcell","name_pattern":".*","action_text":"Table text"}}
{{"action_type":"keyboard","role":"","name_pattern":"","action_text":"Enter"}}
{{"action_type":"noop","role":"","name_pattern":"","action_text":""}}

Available elements:
{elements_context}"""

    
    # Send image to GPT-4o for visual understanding
    try:
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
                        "text": f"""Task: {state["goal"]}
{failed_context}
{actions_context}

Elements:
{elements_context}

Rules:
1. If task needs Settings/Theme/Preferences AND you see [link] Log in → CLICK LOGIN FIRST (settings are only available after login!)
2. If task needs Theme/Appearance AND you see [menuitem] with user's name → CLICK IT (user settings are in profile menu!)
3. If you just recovered (pressed Escape) → Try HOVER to explore OR click a DIFFERENT element (don't repeat the same action!)
4. SKIP column headers (Status, Assignee, Task name, etc.)
5. If you see 🆕 marked buttons (New task, Add task) → CLICK THEM FIRST before typing!
6. HOVER is ONLY for exploring when element is NOT visible (e.g., hover menu button to reveal dropdown)
7. If the element you need IS VISIBLE → CLICK IT (don't hover!)
8. If you see [menuitem] or [option] → dropdown is open, click it
9. If you see [textbox] and no 🆕 buttons → type directly
10. Use EXACT role from list (e.g., [menuitem] → "menuitem")
11. Return noop only when task is 100% complete

Return JSON only:"""
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
            max_tokens=1000
        )
    except Exception as api_error:
        print(f"❌ GPT-4 API Error: {api_error}")
        state["role"] = ""
        state["name_pattern"] = ""
        return state
    
    # Parse response
    message = response.choices[0].message
    raw_content = message.content
    refusal = getattr(message, 'refusal', None)
    
    if refusal:
        print(f"⚠️  GPT-4 Vision REFUSED: {refusal}")
        print("💡 Using smart fallback based on goal and visible elements...")
        
        # Heuristic fallback based on goal keywords
        goal_lower = state["goal"].lower()
        
        # Priority 1: If we see a Login button, prioritize it (most tasks require authentication)
        # This is general - if there's a login available, we're likely on a marketing/public page
        login_el = next((el for el in visible_elements if "log" in el.get("name", "").lower() and "in" in el.get("name", "").lower()), None)
        if not login_el:
            login_el = next((el for el in visible_elements if "sign" in el.get("name", "").lower() and "in" in el.get("name", "").lower()), None)
        
        if login_el:
            print(f"🎯 Fallback: Login available - clicking to access workspace")
            print(f"   Clicking: [{login_el['role']}] {login_el['name']}")
            state["action_type"] = "click"
            state["role"] = login_el['role']
            state["name_pattern"] = "log.*in|sign.*in"
            state["action_text"] = ""
            return state
        
        # Priority 2: Use GPT semantic matching to find the best element
        # This understands synonyms, platform-specific naming, and context
        print("🧠 Using GPT semantic matching to find best element...")
        semantic_match = find_semantic_match(state["goal"], visible_elements)
        
        if semantic_match:
            print(f"🎯 Fallback: GPT found semantically matching element")
            print(f"   Element: [{semantic_match['role']}] {semantic_match['name']}")
            state["action_type"] = "click"
            state["role"] = semantic_match['role']
            # Use first word or full name as pattern
            el_name = semantic_match['name']
            state["name_pattern"] = re.escape(el_name.split()[0]) if el_name and ' ' in el_name else re.escape(el_name) if el_name else ".*"
            state["action_text"] = ""
            return state
        
        # Priority 3: If there's any textbox visible, consider typing into it
        # BUT ONLY if we don't need to navigate to a specific page first
        textbox_el = next((el for el in visible_elements if el.get("role") == "textbox"), None)
        if textbox_el:
            # Check if goal mentions navigating to a specific location (e.g., "on my X", "in my X")
            # Extract all treeitems (navigation targets) from sidebar
            treeitems = [el.get("name", "") for el in visible_elements if el.get("role") == "treeitem"]
            
            # Check if any treeitem name appears in the goal (indicating we need to navigate there)
            needs_navigation = False
            target_page = None
            for treeitem_name in treeitems:
                if len(treeitem_name) > 2:  # Skip very short names
                    # Check if treeitem name appears in goal (case-insensitive, flexible matching)
                    normalized_name = treeitem_name.lower().replace(" ", "").replace("-", "")
                    normalized_goal = goal_lower.replace(" ", "").replace("-", "")
                    
                    if normalized_name in normalized_goal:
                        needs_navigation = True
                        target_page = treeitem_name
                        print(f"🔍 Detected navigation intent: goal mentions '{target_page}' which exists in sidebar")
                        break
            
            # Also check for phrases like "on my", "in my", "to my" which suggest navigation
            if not needs_navigation and treeitems:
                navigation_phrases = [r"on\s+(my\s+)?(\w+)", r"in\s+(my\s+)?(\w+)", r"to\s+(my\s+)?(\w+)"]
                import re as regex_module
                for phrase in navigation_phrases:
                    match = regex_module.search(phrase, goal_lower)
                    if match:
                        target_word = match.group(2) if match.lastindex >= 2 else match.group(1)
                        # Check if this word matches any treeitem
                        for treeitem_name in treeitems:
                            if target_word in treeitem_name.lower():
                                needs_navigation = True
                                target_page = treeitem_name
                                print(f"🔍 Detected navigation phrase: '{match.group(0)}' → likely refers to '{target_page}'")
                                break
                        if needs_navigation:
                            break
            
            # Check if we're already on the right page (task-related buttons visible)
            has_creation_button = any(
                "🆕" in el.get("name", "") or 
                ("new" in el.get("name", "").lower() and any(kw in el.get("name", "").lower() for kw in ["task", "entry", "item", "row"]))
                for el in visible_elements
            )
            
            # Only auto-type if we DON'T need navigation OR we're already on the target page (creation button visible)
            if not needs_navigation or has_creation_button:
                print(f"🎯 Fallback: Typing into available textbox: [{textbox_el['role']}] {textbox_el['name']}")
                # Extract text to type from goal using improved logic
                import re as regex_module
                
                patterns = [
                    r"following\s+in\s+it\s*:\s*(.+?)\.?$",  # "following in it : TEXT"
                    r"in\s+it\s*:\s*(.+?)\.?$",  # "in it : TEXT"
                    r"write\s+the\s+following\s+in\s+it\s*:\s*(.+?)\.?$",
                    r"following[:\s]+(.+?)(?:\s+in\s+it)?\.?$",
                    r"write\s+(.+?)(?:\s+in\s+it)?\.?$",
                    r"content[:\s]+(.+?)\.?$",
                ]
                
                text_to_type = None
                for pattern in patterns:
                    match = regex_module.search(pattern, state["goal"], regex_module.IGNORECASE)
                    if match:
                        text_to_type = match.group(1).strip()
                        # Clean up trailing "in it" or "in the entry"
                        text_to_type = regex_module.sub(r'\s+in\s+(it|the\s+entry)\.?$', '', text_to_type, flags=regex_module.IGNORECASE)
                        break
                
                if not text_to_type:
                    text_to_type = "New entry"
                
                print(f"  → Extracted text: '{text_to_type}'")
                
                state["action_type"] = "type"
                state["role"] = "textbox"
                state["name_pattern"] = ".*"  # Match any textbox
                state["action_text"] = text_to_type
                return state
            else:
                print(f"⚠️  Navigation needed to '{target_page}' first - letting GPT decide")
        
        # Priority 4: If elements list is sparse, try hovering to reveal more UI
        # General approach: hover over any button/link we haven't explored yet
        hover_explored = state.get("hover_explored", [])
        if len(visible_elements) < 15:  # Sparse UI - might have hidden elements
            # Get all clickable elements (buttons, links) - general approach
            hover_candidates = [
                el for el in visible_elements 
                if el.get("role") in ["button", "link"]
            ]
            
            # Filter out already explored ones and try each
            for candidate in hover_candidates:
                hover_key = f"{candidate['role']}:{candidate['name']}"
                if hover_key not in hover_explored:
                    print(f"🔍 Fallback: UI is sparse ({len(visible_elements)} elements) - hovering to explore")
                    print(f"   Hovering: [{candidate['role']}] {candidate['name']}")
                    state["action_type"] = "hover"
                    state["role"] = candidate['role']
                    state["name_pattern"] = re.escape(candidate['name'])  # Exact match
                    state["action_text"] = ""
                    return state
        
        # No fallback available
        print("❌ No fallback action available")
        state["role"] = ""
        state["name_pattern"] = ""
        return state
    
    if not raw_content:
        print("⚠️  GPT-4 Vision returned empty response (no refusal, just empty)")
        state["role"] = ""
        state["name_pattern"] = ""
        return state
    
    try:
        analysis_json = json.loads(raw_content)
        action_type = analysis_json.get("action_type", "click").lower()
        role = analysis_json.get("role", "")
        name_pattern = analysis_json.get("name_pattern", "")
        action_text = analysis_json.get("action_text", "")
        
        # OVERRIDE 1: If GPT chose noop but login is visible (likely on landing page)
        if action_type == "noop":
            login_el = next((el for el in visible_elements if ("log" in el.get("name", "").lower() and "in" in el.get("name", "").lower()) or ("sign" in el.get("name", "").lower() and "in" in el.get("name", "").lower())), None)
            if login_el:
                print("⚠️  GPT chose noop but login is available! Overriding to login first...")
                action_type = "click"
                role = login_el['role']
                name_pattern = "log.*in|sign.*in"
                action_text = ""
        
        # OVERRIDE 2: If GPT is looking for non-existent textbox but there IS a textbox available
        goal_lower = state["goal"].lower()
        if action_type == "type" and ("write" in goal_lower or "entry" in goal_lower):
            # Check if the pattern will actually find something
            textbox_found = any(el.get("role") == "textbox" and re.search(name_pattern, el.get("name", ""), re.IGNORECASE) for el in visible_elements)
            if not textbox_found:
                # No textbox matches the pattern, but there is A textbox
                any_textbox = next((el for el in visible_elements if el.get("role") == "textbox"), None)
                if any_textbox:
                    print(f"⚠️  No textbox matches '{name_pattern}', using available textbox: {any_textbox['name']}")
                    name_pattern = ".*"  # Match any textbox
        
        # OVERRIDE 3: If we just created a page/entry and there's a textbox, TYPE into it!
        # Detect creation actions: clicked a "new" button and now there's a textbox
        just_created_something = any(
            "click:button" in act and any(kw in act for kw in ["new", "add", "create"])
            for act in actions_performed[-2:]  # Check last 2 actions
        )
        
        needs_to_type = any(keyword in goal_lower for keyword in ["write", "following", "name", "title", "call", "label", "add"])
        
        if action_type == "click" and just_created_something and needs_to_type:
            # Check if there's a textbox available and we haven't typed yet
            any_textbox = next((el for el in visible_elements if el.get("role") == "textbox"), None)
            typed_already = any("type:" in act for act in actions_performed)
            
            if any_textbox and not typed_already:
                print(f"⚠️  Just created entry! Switching to TYPE mode into: {any_textbox['name']}")
                # Extract text from goal - look for text after various keywords
                import re as regex_module
                
                # Try multiple patterns to extract the actual content to type
                patterns = [
                    r"name\s+(?:the\s+)?(?:task|entry|page|item)\s+as[:\s]+(.+?)\.?$",  # "name the task as: X"
                    r"name\s+it[:\s]+(.+?)\.?$",  # "name it: X"
                    r"title[:\s]+(.+?)\.?$",  # "title: X"
                    r"call\s+it[:\s]+(.+?)\.?$",  # "call it: X"
                    r"following[:\s]+(.+?)(?:\s+in\s+it)?\.?$",  # "write the following: X"
                    r"write\s+the\s+following[:\s]+(.+?)\.?$",   # "write the following X"
                    r"write\s+(.+?)(?:\s+in\s+it)?\.?$",         # "write X" or "write X in it"
                    r"content[:\s]+(.+?)\.?$",                   # "content: X"
                    r"add\s+(.+?)(?:\s+as\s+(?:a\s+)?(?:task|entry|item))?\.?$",  # "add X" or "add X as task"
                ]
                
                text_to_type = None
                for pattern in patterns:
                    match = regex_module.search(pattern, state["goal"], regex_module.IGNORECASE)
                    if match:
                        text_to_type = match.group(1).strip()
                        # Clean up common trailing phrases
                        text_to_type = regex_module.sub(r'\s+in\s+(it|the\s+entry)\.?$', '', text_to_type, flags=regex_module.IGNORECASE)
                        text_to_type = regex_module.sub(r'\s+as\s+(a\s+)?(task|entry|item)\.?$', '', text_to_type, flags=regex_module.IGNORECASE)
                        break
                
                if not text_to_type:
                    # Fallback: look for quoted text or text after "as:"
                    quote_match = regex_module.search(r'["\'](.+?)["\']', state["goal"])
                    if quote_match:
                        text_to_type = quote_match.group(1)
                    else:
                        text_to_type = "New entry"
                
                action_type = "type"
                role = "textbox"
                name_pattern = ".*"
                action_text = text_to_type
                print(f"  → Will type: '{text_to_type}'")
        
        print("\n" + "="*70)
        print("🤖 GPT-4 Vision Decision:")
        print("="*70)
        print(f"Action Type: {action_type}")
        print(f"Role: {role}")
        print(f"Name Pattern: {name_pattern}")
        if action_type == "type":
            print(f"Text to Type: {action_text}")
        print("="*70 + "\n")
        
        state["action_type"] = action_type
        state["role"] = role
        state["name_pattern"] = name_pattern
        state["action_text"] = action_text
        
    except json.JSONDecodeError as e:
        print(f"⚠️  Failed to parse JSON: {e}")
        print(f"Raw response: {raw_content}")
        # Fallback
        state["action_type"] = "noop"
        state["role"] = ""
        state["name_pattern"] = ""
        state["action_text"] = ""
    
    return state

def executor(state: AgentState) -> AgentState:
    """Execute the action using role-based locators"""
    global i
    print("🤖 Executor: Executing the action...")
    page = get_page()
    i+=1
    global screenshots_dir
    step_screenshot = screenshots_dir / f"step_{i}.png"
    
    # Note: We'll take screenshot AFTER positioning cursor on target element
    # (moved below to show cursor position)
    
    role = state.get("role", "").strip()
    name_pattern = state.get("name_pattern", "").strip()
    action_type = state.get("action_type", "click").lower()
    action_text = state.get("action_text", "")
    
    # Keyboard and noop actions don't need role/pattern
    if action_type not in ["keyboard", "noop"] and (not role or not name_pattern):
        print("⚠️  No role/pattern provided, skipping action")
        return state
    
    # OPTIMIZATION 1: Check if this exact action was already done
    action_key = f"{action_type}:{role}:{name_pattern}:{action_text}"
    actions_performed = state.get("actions_performed", [])
    failed_actions = state.get("failed_actions", [])
    current_url = page.url
    last_url = state.get("last_url", "")
    
    # Loop detection: same action already performed
    if action_key in actions_performed:
        print(f"⚠️  LOOP DETECTED: Already performed {action_key}")
        
        # Check if goal was already achieved (e.g., toggle already in desired state)
        already_complete = any(":already_complete" in act for act in actions_performed)
        goal_achieved = state.get("goal_text_entered", False)
        
        if already_complete or goal_achieved:
            print("   ✅ Goal already achieved - TERMINATING (not pressing Escape)")
            state["role"] = ""
            state["name_pattern"] = ""
            return state
        
        # Count how many recovery attempts we've done
        recovery_count = sum(1 for act in actions_performed if act.startswith("recovery:escape"))
        
        if recovery_count >= 3:
            print(f"   ⚠️  Too many recovery attempts ({recovery_count}) - TERMINATING")
            print("   Agent may be stuck or goal already achieved without detection")
            state["role"] = ""
            state["name_pattern"] = ""
            return state
        
        # Recovery strategy: Close wrong modal/menu and let GPT try again
        print(f"   🔄 Recovery attempt {recovery_count + 1}/3 - pressing Escape to close modal/menu...")
        try:
            page.keyboard.press("Escape")
            page.wait_for_timeout(800)
            print("   ✓ Pressed Escape - modal/menu closed, will explore elsewhere")
            
            # Mark this as a recovery attempt, don't stop execution
            # Let the next cycle explore with hover or try a different element
            state.setdefault("actions_performed", []).append(f"recovery:escape:from:{action_key}")
            return state
        except Exception as e:
            print(f"   ⚠️  Escape failed: {e}")
            state["role"] = ""
            state["name_pattern"] = ""
            return state
    
    # Skip known failed actions (but allow retries after recovery)
    if action_key in failed_actions:
        # Check if we just did a recovery - if so, allow retry
        recent_recovery = any("recovery:escape" in act for act in actions_performed[-2:])
        if not recent_recovery:
            print(f"❌ SKIP: Action already failed {action_key}")
            state["role"] = ""
            state["name_pattern"] = ""
            return state
        else:
            print(f"🔄 Retrying after recovery: {action_key}")
    
    # OPTIMIZATION 2: Detect when stuck (URL not changing after multiple actions)
    if len(actions_performed) >= 3 and current_url == last_url:
        recent_actions = actions_performed[-3:]
        # Ignore recovery actions in this check
        non_recovery_actions = [a for a in recent_actions if not a.startswith("recovery:")]
        if len(non_recovery_actions) >= 2 and len(set(non_recovery_actions)) == 1:
            print(f"⚠️  STUCK: Same action repeated without progress")
            print(f"   🔄 Recovering: Pressing Escape and trying hover exploration...")
            try:
                page.keyboard.press("Escape")
                page.wait_for_timeout(800)
                print("   ✓ Escape pressed - will explore with hover next")
                state.setdefault("actions_performed", []).append("recovery:escape:stuck")
                return state
            except:
                state["role"] = ""
                state["name_pattern"] = ""
                return state
    
    state["last_url"] = current_url
    
    try:
        # Handle keyboard actions first (don't need to locate element)
        if action_type == "keyboard":
            # Keyboard action: press a key
            print(f"⌨️  Pressing key: {action_text}")
            page.keyboard.press(action_text)
            page.wait_for_timeout(1000)
            print(f"✓ Key pressed successfully")
            
            # Track successful action
            state.setdefault("actions_performed", []).append(action_key)
        
        elif action_type == "hover":
            # Hover action: hover over element to reveal hidden UI
            print(f"👆 Hovering over [{role}] matching pattern: {name_pattern}")
            
            # Locate element
            loc = page.get_by_role(role, name=re.compile(name_pattern, re.IGNORECASE)).first
            
            try:
                loc.wait_for(state="visible", timeout=5000)
                
                # Position cursor and add visual marker before hovering
                try:
                    bbox = loc.bounding_box()
                    if bbox:
                        center_x = bbox['x'] + bbox['width'] / 2
                        center_y = bbox['y'] + bbox['height'] / 2
                        
                        # Add visual cursor marker (blue for hover)
                        page.evaluate(f"""
                            const marker = document.createElement('div');
                            marker.id = 'cursor-marker-temp';
                            marker.style.cssText = `
                                position: fixed;
                                left: {center_x}px;
                                top: {center_y}px;
                                width: 24px;
                                height: 24px;
                                border: 3px solid blue;
                                border-radius: 50%;
                                background: rgba(0, 100, 255, 0.2);
                                pointer-events: none;
                                z-index: 999999;
                                transform: translate(-50%, -50%);
                                animation: pulse 0.5s ease-in-out infinite;
                            `;
                            const style = document.createElement('style');
                            style.textContent = '@keyframes pulse {{ 0%, 100% {{ opacity: 1; transform: translate(-50%, -50%) scale(1); }} 50% {{ opacity: 0.6; transform: translate(-50%, -50%) scale(1.2); }} }}';
                            document.head.appendChild(style);
                            document.body.appendChild(marker);
                        """)
                        
                        page.mouse.move(center_x, center_y)
                        page.wait_for_timeout(400)
                        print(f"🎯 Hover cursor at ({int(center_x)}, {int(center_y)})")
                        
                        # Take screenshot with cursor marker
                        page.screenshot(path=str(step_screenshot))
                        print(f"📸 Step {i} screenshot (hover marker): {step_screenshot}")
                        
                        # Remove marker
                        page.evaluate("document.getElementById('cursor-marker-temp')?.remove()")
                except:
                    pass
                
                print(f"⏳ Hovering to reveal hidden elements...")
                loc.hover(timeout=3000)
                page.wait_for_timeout(1500)  # Wait for any animations or dropdowns to appear
                print(f"✓ Hover successful - checking for new elements...")
                
                # Track that we hovered over this element
                hover_key = f"{role}:{name_pattern}"
                state.setdefault("hover_explored", []).append(hover_key)
                state.setdefault("actions_performed", []).append(action_key)
                
                # Note: inspector will re-scan after this and report any new elements
                
            except Exception as hover_err:
                print(f"❌ Hover failed: {hover_err}")
                state.setdefault("failed_actions", []).append(action_key)
        
        else:
            # For click/type actions, we need to locate the element first
            # Wait a bit for any modals/animations to settle
            page.wait_for_timeout(800)
            
            print(f"🔍 Looking for [{role}] matching pattern: {name_pattern}")
            
            # Use Playwright's role-based locator with regex (case-insensitive)
            loc = page.get_by_role(role, name=re.compile(name_pattern, re.IGNORECASE)).first
            
            # Wait for element to be visible
            print(f"⏳ Waiting for element to be visible...")
            try:
                loc.wait_for(state="visible", timeout=5000)
                
                # Get element position and add visual cursor marker
                try:
                    bbox = loc.bounding_box()
                    if bbox:
                        # Calculate center position
                        center_x = bbox['x'] + bbox['width'] / 2
                        center_y = bbox['y'] + bbox['height'] / 2
                        
                        # Add visual cursor marker (red pulsing circle)
                        page.evaluate(f"""
                            const marker = document.createElement('div');
                            marker.id = 'cursor-marker-temp';
                            marker.style.cssText = `
                                position: fixed;
                                left: {center_x}px;
                                top: {center_y}px;
                                width: 24px;
                                height: 24px;
                                border: 3px solid red;
                                border-radius: 50%;
                                background: rgba(255, 0, 0, 0.2);
                                pointer-events: none;
                                z-index: 999999;
                                transform: translate(-50%, -50%);
                                animation: pulse 0.5s ease-in-out infinite;
                            `;
                            const style = document.createElement('style');
                            style.textContent = '@keyframes pulse {{ 0%, 100% {{ opacity: 1; transform: translate(-50%, -50%) scale(1); }} 50% {{ opacity: 0.6; transform: translate(-50%, -50%) scale(1.2); }} }}';
                            document.head.appendChild(style);
                            document.body.appendChild(marker);
                        """)
                        
                        # Move actual cursor too
                        page.mouse.move(center_x, center_y)
                        page.wait_for_timeout(400)  # Let animation show
                        print(f"🎯 Cursor marker at ({int(center_x)}, {int(center_y)})")
                except:
                    pass
                
                # NOW take screenshot with visual cursor marker
                page.screenshot(path=str(step_screenshot))
                print(f"📸 Step {i} screenshot (with cursor marker): {step_screenshot}")
                
                # Remove cursor marker before clicking
                try:
                    page.evaluate("document.getElementById('cursor-marker-temp')?.remove()")
                except:
                    pass
            except Exception as wait_err:
                # Element not found - ask GPT Vision for better regex
                print(f"⚠️  Element not found: {wait_err}")
                print("🤔 Asking GPT Vision to analyze screenshot for better pattern...")
                
                better_pattern = ask_gpt_for_better_regex(
                    goal=state["goal"],
                    failed_pattern=name_pattern,
                    img_base64=state["img_base64"],
                    visible_elements=state.get("visible_elements", [])
                )
                
                if better_pattern and better_pattern != name_pattern:
                    print(f"🔄 Retrying with new pattern: {better_pattern}")
                    loc = page.get_by_role(role, name=re.compile(better_pattern, re.IGNORECASE)).first
                    loc.wait_for(state="visible", timeout=5000)
                else:
                    raise wait_err  # Re-raise original error
            
            if action_type == "type":
                # Type action: fill input field
                print(f"📝 Typing '{action_text}' into [{role}]...")
                
                # FIRST: Check if there's a blocking modal - close it before typing
                try:
                    modal_present = page.locator("[role='dialog']:visible, .notion-overlay-container:visible").count() > 0
                    if modal_present:
                        print("  ⚠️  Modal detected - closing it before typing...")
                        page.keyboard.press("Escape")
                        page.wait_for_timeout(500)
                        print("  ✓ Modal closed")
                except:
                    pass
                
                # Add visual marker on the input field before typing
                try:
                    bbox = loc.bounding_box()
                    if bbox:
                        center_x = bbox['x'] + bbox['width'] / 2
                        center_y = bbox['y'] + bbox['height'] / 2
                        
                        # Add green cursor marker for typing
                        page.evaluate(f"""
                            const marker = document.createElement('div');
                            marker.id = 'cursor-marker-temp';
                            marker.style.cssText = `
                                position: fixed;
                                left: {center_x}px;
                                top: {center_y}px;
                                width: 20px;
                                height: 20px;
                                border: 3px solid green;
                                border-radius: 50%;
                                background: rgba(0, 255, 0, 0.2);
                                pointer-events: none;
                                z-index: 999999;
                                transform: translate(-50%, -50%);
                            `;
                            document.body.appendChild(marker);
                        """)
                        page.wait_for_timeout(300)
                        print(f"⌨️  Type cursor at ({int(center_x)}, {int(center_y)})")
                except:
                    pass
                
                # For gridcells or cells, click first to activate
                if role in ["cell", "gridcell"]:
                    print("  (table cell - clicking to activate)")
                    loc.click(timeout=3000)
                    page.wait_for_timeout(300)
                    # After clicking cell, type directly
                    page.keyboard.type(action_text, delay=50)
                else:
                    # Check if it's a contenteditable element
                    is_contenteditable = loc.evaluate("el => el.contentEditable === 'true'")
                    
                    if is_contenteditable:
                        print("  (contenteditable element - using keyboard)")
                        loc.click(timeout=3000)
                        page.keyboard.press("Meta+A")  # Select all
                        page.keyboard.press("Backspace")  # Clear
                        page.keyboard.type(action_text, delay=50)
                    else:
                        # Regular input/textarea
                        print("  (regular input field)")
                        loc.click(timeout=3000)
                        loc.fill(action_text)
                
                page.wait_for_timeout(500)
                print(f"✓ Typed '{action_text}' successfully")
                
                # Remove cursor marker
                try:
                    page.evaluate("document.getElementById('cursor-marker-temp')?.remove()")
                except:
                    pass
                
                # Track successful action
                state.setdefault("actions_performed", []).append(action_key)
                
                # OPTIMIZATION 3: Check if goal text was typed
                goal_lower = state.get("goal", "").lower()
                if action_text.lower() in goal_lower or any(word in action_text.lower() for word in goal_lower.split() if len(word) > 3):
                    print(f"🎯 Goal text '{action_text}' entered! Marking complete.")
                    state["goal_text_entered"] = True
            
            else:
                # Click action (default)
                print(f"🖱️  Clicking [{role}] matching '{name_pattern}'...")
                
                # Special handling for switches/toggles
                if role == "switch":
                    # Get current state before clicking
                    aria_checked_before = loc.get_attribute("aria-checked")
                    print(f"  🔘 Toggle state BEFORE: {aria_checked_before}")
                    
                    element_description = name_pattern if name_pattern else "toggle switch"
                    goal_already_met = False
                    
                    # STEP 1: Check current state with CLEAR information
                    if aria_checked_before and aria_checked_before in ["true", "false"]:
                        # We have clear aria-checked state - use text-based GPT check
                        state_description = f"Toggle is {'ON' if aria_checked_before == 'true' else 'OFF'}"
                        goal_already_met = check_goal_achieved_by_state(
                            goal=state.get("goal", ""),
                            element_name=element_description,
                            current_state=state_description
                        )
                    else:
                        # STEP 2: aria-checked is None/unclear - use GPT VISION to inspect screenshot
                        print(f"  👁️  aria-checked is unclear, using GPT Vision to analyze screenshot...")
                        goal_already_met, visual_state = check_toggle_state_from_screenshot(
                            goal=state.get("goal", ""),
                            element_name=element_description,
                            img_base64=state.get("img_base64", "")
                        )
                    
                    # STEP 3: Decide whether to click
                    if goal_already_met:
                        print(f"✅ Goal already achieved - but CLICKING ANYWAY for demonstration!")
                        # For demo purposes, we'll click to show the action
                        # Comment out the next 3 lines if you want to skip clicking when goal is met
                        # state["goal_text_entered"] = True
                        # state.setdefault("actions_performed", []).append(action_key + ":already_complete")
                        # return state
                    
                    # STEP 4: Click the toggle (always for demo, or when goal not met)
                    print(f"  🖱️  Clicking toggle to {'demonstrate action' if goal_already_met else 'achieve goal'}...")
                    loc.click(timeout=5000)
                    page.wait_for_timeout(1200)  # Wait for animation
                    
                    # STEP 5: Verify state after clicking
                    aria_checked_after = loc.get_attribute("aria-checked")
                    print(f"  🔘 Toggle state AFTER: {aria_checked_after}")
                    
                    # STEP 6: Check if goal is NOW achieved
                    if aria_checked_after and aria_checked_after in ["true", "false"]:
                        state_description_after = f"Toggle is {'ON' if aria_checked_after == 'true' else 'OFF'}"
                        goal_now_met = check_goal_achieved_by_state(
                            goal=state.get("goal", ""),
                            element_name=element_description,
                            current_state=state_description_after
                        )
                    else:
                        # Use vision again to verify
                        print(f"  👁️  Verifying final state with GPT Vision...")
                        goal_now_met, visual_state_after = check_toggle_state_from_screenshot(
                            goal=state.get("goal", ""),
                            element_name=element_description,
                            img_base64=state.get("img_base64", "")
                        )
                    
                    if goal_now_met or goal_already_met:
                        print("🎯 Toggle goal achieved! Marking complete.")
                        state["goal_text_entered"] = True
                    else:
                        print("⚠️  Clicked toggle but goal may not be achieved - continuing...")
                
                elif role in ["option", "menuitem", "menuitemradio", "menuitemcheckbox"]:
                    # Dropdown option or menu item
                    print(f"  📋 Clicking dropdown/menu option...")
                    loc.click(timeout=5000)
                    page.wait_for_timeout(800)
                    print(f"✓ Option selected!")
                    
                    # Use GPT to check if this selection achieves the goal (GENERAL!)
                    element_description = name_pattern if name_pattern else "menu option"
                    state_description = f"Selected '{element_description}'"
                    
                    goal_achieved = check_goal_achieved_by_state(
                        goal=state.get("goal", ""),
                        element_name=element_description,
                        current_state=state_description
                    )
                    
                    if goal_achieved:
                        print("🎯 Menu option achieves goal! Marking complete.")
                        state["goal_text_entered"] = True
                
                else:
                    # Regular click (button, link, tab, etc.)
                    loc.click(timeout=5000)
                    page.wait_for_timeout(1500)  # Wait for navigation/modal
                    print(f"✓ Click successful! Current URL: {page.url}")
                    
                    # Check if a modal opened
                    modal_visible = page.locator("[role='dialog']:visible, [role='alertdialog']:visible, [class*='modal']:visible").count() > 0
                    if modal_visible:
                        print("  ℹ️  Modal/dialog detected after click")
                
                # Track successful action
                state.setdefault("actions_performed", []).append(action_key)
            
    except Exception as e:
        error_msg = str(e)[:300]
        print(f"❌ Action failed: {error_msg}")
        
        # Track failed action
        state.setdefault("failed_actions", []).append(action_key)
        print(f"⚠️  Added to failed actions: {action_key}")
    
    return state

def set_goal(state: AgentState) -> AgentState:
    """Set the goal and website URL"""
    global screenshots_dir
    
    # Get task name for folder organization
    task_name = input("Enter a name for this task (used for screenshot folder): ").strip()
    if not task_name:
        task_name = "untitled_task"
    
    # Sanitize folder name (remove special characters)
    import re
    safe_name = re.sub(r'[^\w\s-]', '', task_name).strip().replace(' ', '_').lower()
    
    # Create task-specific screenshot folder
    from pathlib import Path
    screenshots_dir = Path(f"screenshots/{safe_name}")
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Screenshots will be saved to: {screenshots_dir}/")
    
    # Get URL and goal
    url = input("Enter the website URL (e.g., https://www.notion.com): ").strip()
    
    # Add https:// if missing
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
        print(f"  → Auto-corrected to: {url}")
    
    state["website_url"] = url
    state["goal"] = input("Enter the goal of the agent: ")
    return state
def decide_next_action(state: AgentState) -> str:
    """Decide the next action based on the state"""
    if state.get("role", "") == "" or state.get("name_pattern", "") == "":
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
            "messages": [HumanMessage(content="Navigate and accomplish the goal")],
            "screenshot": "",
            "img_base64": "",
            "goal": "",
            "website_url": "",
            "role": "",
            "name_pattern": "",
            "action_type": "click",
            "action_text": "",
            "visible_elements": [],
            "is_first_visit": True,
            "actions_performed": [],
            "failed_actions": [],
            "hover_explored": [],
            "goal_text_entered": False,
            "last_url": ""
        }
        
        config = {"recursion_limit": 50}
        final_state = app.invoke(init_state, config)
        
        print("\n" + "="*70)
        print("COMPLETE")
        print("="*70)
        print(f"Goal: {final_state.get('goal')}")
        print(f"Messages exchanged: {len(final_state['messages'])}")
        print(f"📁 Screenshots saved to: {screenshots_dir}/")
        print(f"   Total steps captured: {i}")
        
    finally:
        # Clean up Playwright resources
        print("\n🧹 Cleaning up...")
        cleanup_browser()