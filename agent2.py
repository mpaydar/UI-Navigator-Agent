from __future__ import annotations

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
def build_state_snapshot(page: Page, state: AgentState, visible_elements: list[dict]) -> dict:
    """Collect a compact, structured JSON snapshot of runtime UI state for GPT."""
    # URL & title
    url = page.url
    try:
        title = page.title()
    except Exception:
        title = ""

    # Modal detection
    active_modal = None
    modal_title = ""
    try:
        modal = page.locator("[role='dialog']:visible,[role='alertdialog']:visible,[class*='modal']:visible").first
        if modal and modal.count() > 0 and modal.is_visible():
            active_modal = True
            # Try to harvest a heading inside modal
            try:
                modal_title = modal.get_by_role("heading").first.inner_text(timeout=300).strip()
            except Exception:
                try:
                    aria_label = modal.get_attribute("aria-label") or ""
                    modal_title = aria_label.strip()
                except Exception:
                    modal_title = ""
        else:
            active_modal = False
    except Exception:
        active_modal = False

    # Focused element (best effort)
    try:
        focus_info = page.evaluate("""
            () => {
              const el = document.activeElement;
              if (!el) return null;
              const role = el.getAttribute?.('role') || '';
              const tag = el.tagName || '';
              const aria = el.getAttribute?.('aria-label') || '';
              const ph = el.getAttribute?.('placeholder') || '';
              const ce = (el as any).isContentEditable === true;
              let txt = '';
              try { txt = (el as any).innerText?.slice(0,80) || ''; } catch(e){}
              return { tag, role, aria, placeholder: ph, isContentEditable: ce, previewText: txt };
            }
        """)
    except Exception:
        focus_info = None

    # Scroll/viewport
    try:
        scroll = page.evaluate("""
            () => ({
                x: Math.round(window.scrollX),
                y: Math.round(window.scrollY),
                height: document.documentElement.scrollHeight || 0
            })
        """)
    except Exception:
        scroll = {"x": 0, "y": 0, "height": 0}

    # Heuristic login state (non-invasive)
    def _has(patterns):
        for el in visible_elements:
            nm = (el.get("name") or "").lower()
            if any(p in nm for p in patterns):
                return True
        return False

    is_login_page = bool(re.search(r'/(login|signin|sign-in|auth|register)(/|$)', url.split("?")[0], re.IGNORECASE))
    login_cta_visible = _has(["log in", "signin", "sign in"])
    logout_cta_visible = _has(["log out", "sign out", "logout"])
    is_logged_in = (not is_login_page) and (logout_cta_visible or not login_cta_visible)

    # Diff visible elements vs last turn (tiny)
    last_vis = state.get("last_visible_elements", [])
    last_set = {(e.get("role"), e.get("name")) for e in last_vis}
    cur_set = {(e.get("role"), e.get("name")) for e in visible_elements}
    added = list(cur_set - last_set)
    removed = list(last_set - cur_set)

    snapshot = {
        "url": url,
        "title": title,
        "page_context": "workspace" if is_logged_in else "pre-login",
        "active_modal": active_modal,
        "active_modal_title": modal_title,
        "focus_element": focus_info or None,
        "scroll": scroll,
        "variables": {
            "is_logged_in": is_logged_in,
            "login_cta_visible": login_cta_visible,
            "logout_cta_visible": logout_cta_visible
        },
        "visible_elements": visible_elements[:40],  # cap to keep prompt lean
        "recent_actions": state.get("actions_performed", [])[-10:],
        "failed_actions": state.get("failed_actions", [])[-10:],
        "diff": {
            "added": [{"role": r, "name": n} for (r, n) in added][:15],
            "removed": [{"role": r, "name": n} for (r, n) in removed][:15]
        }
    }
    return snapshot


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

IMPORTANT CONTEXT:
- In modern web apps (Notion, Linear, etc.), creating and naming an entry is OFTEN A SINGLE ACTION
- If the goal is "create X and name it Y", typing "Y" into a NEW entry's title field COMPLETES BOTH STEPS
- Typing a title/name into a newly created page/task/database IS the creation - no "submit" button needed
- Auto-save is standard - typing and moving on = creation complete
- For invitation/application workflows, clicking "Send invite"/"Apply"/"Submit" COMPLETES the action
- Look for evidence of completion (e.g., member count increased, confirmation message, modal closed)

DECISION RULES:
1. If goal = "create [something] and name/call it [X]" AND current state = "Typed '[X]' into textbox"
   → goal_satisfied = TRUE (creation + naming complete in one action!)
2. If goal = "invite people" AND current state = "Clicked button 'Send invite'"
   → goal_satisfied = TRUE (invitation sent!)
3. If goal = "apply for job" AND current state = "Clicked button 'Submit application'"
   → goal_satisfied = TRUE (application submitted!)
4. If goal has additional DISTINCT steps (e.g., "create X, name it Y, THEN add description")
   → Check if current state completed ALL steps, not just first
5. If goal mentions a CONTEXT ("from settings", "via menu") but the ACTION is complete
   → goal_satisfied = TRUE (context is just navigation, not the goal itself!)

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

def _append_unique(visible_elements, role, name):
    """Append one element if it's not already present."""
    if not name:
        return
    if not any(el['role'] == role and el['name'] == name for el in visible_elements):
        visible_elements.append({
            "role": role,
            "name": name,
            "description": f"{role} '{name}'"
        })

def inspector(state: AgentState) -> AgentState:
    """Navigate to website and take screenshot (now also builds structured runtime snapshot)."""
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
    url_path = current_url.split('?')[0]  # Remove query params
    login_pattern = re.compile(r'/(login|signin|sign-in|signup|sign-up|auth|authenticate|register)(/|$)', re.IGNORECASE)
    if login_pattern.search(url_path):
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
    
    try:
        tree = page.accessibility.snapshot()
        
        def extract_elements(node, depth=0):
            if not isinstance(node, dict):
                return
            
            role = node.get("role", "")
            name = node.get("name", "")
            disabled = node.get("disabled", False)
            
            interactive_roles = [
                "link", "button", "textbox", "switch", "tab", 
                "option", "menuitem", "menuitemradio", "menuitemcheckbox",
                "treeitem", "combobox", "listbox", "row", "cell", "gridcell",
                "search", "searchbox"
            ]
            
            if disabled:
                return
            
            if role in interactive_roles and name and len(name) < 150:
                skip_patterns = ["hidden properties", "hidden columns", "drag"]
                if any(skip in name.lower() for skip in skip_patterns):
                    return
                
                if not any(el['name'] == name and el['role'] == role for el in visible_elements):
                    is_creation_element = (
                        role in ["button", "link"] and 
                        any(keyword in name.lower() for keyword in ["+ new", "new page", "+ add"])
                    )
                    if is_creation_element:
                        existing_similar = sum(1 for el in visible_elements 
                                              if el['name'].lower() == name.lower())
                        if existing_similar >= 2:
                            return
                    if role == "button" and ("new" in name.lower() and ("task" in name.lower() or "to-do" in name.lower() or "todo" in name.lower())):
                        name = f"🆕 {name}"
                    if role == "menuitem" and ("'s" in name or "profile" in name.lower() or "account" in name.lower()):
                        name = f"📋{name}"
                    
                    visible_elements.append({
                        "role": role,
                        "name": name,
                        "description": f"{role} '{name}'"
                    })
            
            for child in node.get("children", []):
                if isinstance(child, dict):
                    extract_elements(child, depth + 1)
        
        extract_elements(tree)
        
        if len(visible_elements) == 0:
            print("⚠️  Accessibility tree empty - using direct DOM inspection (SPA detected)...")
            page.wait_for_timeout(2000)
            tree = page.accessibility.snapshot()
            extract_elements(tree)
            
            if len(visible_elements) == 0:
                print("⚠️  Accessibility tree still empty - using comprehensive DOM fallback...")
                page.wait_for_timeout(3000)
                
                try:
                    print("  🔍 Extracting buttons...")
                    buttons = page.locator("button").all()
                    print(f"  → Found {len(buttons)} button elements")
                    
                    buttons_added = 0
                    buttons_skipped = 0
                    for btn in buttons[:100]:
                        try:
                            aria_label = ""
                            text = ""
                            title = ""
                            try: aria_label = btn.get_attribute("aria-label", timeout=300) or ""
                            except: pass
                            try: text = btn.inner_text(timeout=300).strip()
                            except: pass
                            try: title = btn.get_attribute("title", timeout=300) or ""
                            except: pass
                            name = aria_label or text or title
                            if name and len(name) > 1 and len(name) < 300:
                                visible_elements.append({"role": "button", "name": name, "description": f"button '{name}'"})
                                buttons_added += 1
                            else:
                                buttons_skipped += 1
                        except Exception:
                            buttons_skipped += 1
                    
                    print(f"  ✓ Added {buttons_added} buttons to list ({buttons_skipped} skipped)")
                    
                    print(f"  🔍 Searching for action buttons by common patterns...")
                    try:
                        action_button_selectors = [
                            "button[id*='apply']", "button[class*='apply']",
                            "button[id*='submit']", "button[class*='submit']",
                            "button[id*='save']", "button[class*='save']",
                            "button[aria-label*='apply' i]", "button[aria-label*='submit' i]"
                        ]
                        for selector in action_button_selectors:
                            try:
                                action_btn = page.locator(selector).first
                                if action_btn.count() > 0:
                                    aria = action_btn.get_attribute("aria-label", timeout=500) or ""
                                    btn_text = ""
                                    try: btn_text = action_btn.inner_text(timeout=500).strip()
                                    except: pass
                                    btn_name = aria or btn_text
                                    if btn_name and not any(el.get('name') == btn_name for el in visible_elements):
                                        print(f"  🎯 FOUND action button via '{selector}': '{btn_name}'")
                                        visible_elements.append({"role": "button", "name": btn_name, "description": f"button '{btn_name}'"})
                                        buttons_added += 1
                            except: pass
                    except Exception as action_err:
                        print(f"  ⚠️  Action button search failed: {action_err}")
                    
                    print(f"  🔍 Extracting links...")
                    links = page.locator("a").all()
                    print(f"  → Found {len(links)} link elements")
                    links_added = 0
                    for link in links[:100]:
                        try:
                            aria_label = link.get_attribute("aria-label", timeout=200) or ""
                            text = ""
                            try: text = link.inner_text(timeout=200).strip()
                            except: pass
                            name = aria_label or text
                            if name and len(name) > 1 and len(name) < 300:
                                if not any(el.get('name') == name for el in visible_elements):
                                    visible_elements.append({"role": "link", "name": name, "description": f"link '{name}'"})
                                    links_added += 1
                        except: pass
                    print(f"  ✓ Added {links_added} links to list")
                    
                    print(f"  🔍 Extracting inputs...")
                    inputs = page.locator("input, textarea").all()
                    print(f"  → Found {len(inputs)} input elements")
                    inputs_added = 0
                    for inp in inputs[:50]:
                        try:
                            aria_label = inp.get_attribute("aria-label", timeout=200) or ""
                            placeholder = inp.get_attribute("placeholder", timeout=200) or ""
                            name = aria_label or placeholder or "input"
                            if name and len(name) > 0:
                                if not any(el.get('name') == name for el in visible_elements):
                                    visible_elements.append({"role": "textbox", "name": name, "description": f"textbox '{name}'"})
                                    inputs_added += 1
                        except: pass
                    print(f"  ✓ Added {inputs_added} inputs to list")
                    
                    print(f"  🔍 Checking for iframes...")
                    try:
                        frames = page.frames
                        print(f"  → Found {len(frames)} frames")
                        if len(frames) > 1:
                            for frame in frames[1:3]:
                                try:
                                    frame_buttons = frame.locator("button").all()
                                    print(f"    → Iframe has {len(frame_buttons)} buttons")
                                    for btn in frame_buttons[:30]:
                                        try:
                                            aria = btn.get_attribute("aria-label", timeout=200) or ""
                                            txt = ""
                                            try: txt = btn.inner_text(timeout=200).strip()
                                            except: pass
                                            name = aria or txt
                                            if name and "apply" in name.lower():
                                                print(f"    🎯 FOUND in iframe: '{name}'")
                                                visible_elements.append({"role": "button", "name": name, "description": f"button '{name}'"})
                                                buttons_added += 1
                                        except: pass
                                except Exception: pass
                    except Exception as iframe_err:
                        print(f"  ⚠️  Iframe check failed: {iframe_err}")
                except Exception as dom_err:
                    print(f"  ⚠️  DOM extraction error: {dom_err}")
                
                print(f"  📊 Total extracted so far: {len(visible_elements)} elements")
        
        try:
            all_buttons = page.locator("button:visible, div[role='button']:visible, a:visible").all()
            for btn in all_buttons[:50]:
                try:
                    text = btn.inner_text().strip()
                    aria_label = btn.get_attribute("aria-label") or ""
                    display_name = text or aria_label
                    if display_name and len(display_name) < 80:
                        is_task_button = (
                            ("new" in display_name.lower() and "task" in display_name.lower()) or
                            ("add" in display_name.lower() and "task" in display_name.lower()) or
                            ("+" in display_name and "task" in display_name.lower()) or
                            (display_name.lower() == "+ new task") or
                            ("new to-do" in display_name.lower())
                        )
                        is_general_add = "+" in display_name and len(display_name) < 15
                        if (is_task_button or is_general_add) and not any(el['name'] == display_name or el['name'] == f"🆕 {display_name}" for el in visible_elements):
                            visible_elements.append({
                                "role": "button",
                                "name": f"🆕 {display_name}",
                                "description": f"button '{display_name}'"
                            })
                            print(f"  💡 Found via locator: {display_name}")
                except: pass
        except Exception as e:
            print(f"⚠️ Supplementary search failed: {e}")
        
        for i, el in enumerate(visible_elements[:40]):
            role = el['role']
            name = el['name']
            if role == "switch":
                print(f"{i+1}. [🔘{role}] {name}")
            elif role in ["option", "menuitem", "menuitemradio", "combobox"]:
                print(f"{i+1}. [📋{role}] {name}")
            else:
                print(f"{i+1}. [{role}] {name}")
                
    except Exception as e:
        print(f"⚠️ Error extracting from accessibility tree: {e}")
    
    state["visible_elements"] = visible_elements
    
    print(f"\n📊 Total elements extracted: {len(visible_elements)}")
    print(f"   Roles breakdown: {', '.join(set(el['role'] for el in visible_elements))}")
    
    column_buttons = [el for el in visible_elements if el['role'] == 'button' and any(col in el['name'].lower() for col in ['task name', 'status', 'assignee', 'due date', 'priority'])]
    if len(column_buttons) >= 3:
        print(f"\n📋 Table detected with {len(column_buttons)} column headers")
        print("   💡 Hint: Look for 'New' or 'Add' button to create a row, or find a textbox to type directly")
    
    theme_elements = [el for el in visible_elements if "dark" in el['name'].lower() or "light" in el['name'].lower() or "theme" in el['name'].lower()]
    if theme_elements:
        print(f"\n🎨 Theme-related elements found:")
        for el in theme_elements:
            print(f"   [{el['role']}] {el['name']}")
    
    # Screenshot
    global screenshots_dir
    current_screenshot = screenshots_dir / "step_current.png"
    page.screenshot(path=str(current_screenshot))
    with open(current_screenshot, "rb") as f:
        state["img_base64"] = base64.b64encode(f.read()).decode("utf-8")
    print(f"📸 Screenshot saved: {current_screenshot}\n")

    # 🧠 NEW: build and store structured snapshot for GPT
    snapshot = build_state_snapshot(page, state, visible_elements)
    state["snapshot"] = snapshot

    # Keep last elements for diff in next turn
    state["last_visible_elements"] = visible_elements[:]

    return state



def infer_intent(goal: str) -> set[str]:
    g = goal.lower()
    intent = set()
    if re.search(r'\b(setting|settings|preference|preferences|account|profile|billing|workspace)\b', g):
        intent.add("settings")
    if re.search(r'\b(theme|appearance|dark\s*mode|light\s*mode)\b', g):
        intent.add("theme")
    if re.search(r'\b(search|find|filter|look\s*for)\b', g):
        intent.add("search")
    if re.search(r'\b(new|create|add|make)\b', g):
        intent.add("create")
    if re.search(r'\b(page|database|db|table|board|list)\b', g):
        intent.add("content")
    return intent




       
def planner(state: AgentState) -> AgentState:
    """Analyze screenshot + structured runtime snapshot with GPT and plan next action (intent-gated)."""
    print("🤖 Planner: Analyzing screenshot with GPT-4 Vision + state snapshot...")
    page = get_page()
    
    if state.get("goal_text_entered", False):
        print("✅ Goal already complete! Terminating.")
        state["role"] = ""
        state["name_pattern"] = ""
        return state
    
    visible_elements = state.get("visible_elements", [])
    
    # PRE-CHECK: Is the goal already achieved based on visible elements?
    # E.g., "invite user@email.com" and user@email.com is already visible in the MEMBER list
    goal = state.get("goal", "")
    if goal and visible_elements and ("invite" in goal.lower() or "member" in goal.lower()):
        # Only check if we're in a members/settings context
        # Look for tabs like "People", "Members", "Guests" - indicators we're in the right place
        member_context_keywords = ["people", "members", "guests", "contacts", "teamspace settings"]
        in_member_context = any(
            el.get("role") == "tab" and any(kw in el.get("name", "").lower() for kw in member_context_keywords)
            for el in visible_elements
        )
        
        if in_member_context:
            # Extract email addresses from goal
            import re as regex_module
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails_in_goal = regex_module.findall(email_pattern, goal)
            
            if emails_in_goal:
                # Check if any of these emails are visible in the MEMBER context (not just as page names)
                # Look specifically in tab content, not treeitem (which are pages)
                member_elements_text = " ".join([
                    el.get("name", "") for el in visible_elements 
                    if el.get("role") not in ["treeitem", "button"]  # Exclude page/nav elements
                ]).lower()
                
                for email in emails_in_goal:
                    if email.lower() in member_elements_text:
                        print(f"🎯 GOAL ALREADY ACHIEVED: '{email}' is already in the member list!")
                        print(f"   → No action needed, marking complete.")
                        state["goal_text_entered"] = True
                        state["role"] = ""
                        state["name_pattern"] = ""
                        return state
    img_base64 = state.get("img_base64", "")
    snapshot = state.get("snapshot", {})  # <<— NEW
    
    if not img_base64:
        print("❌ No screenshot available!")
        state["role"] = ""
        state["name_pattern"] = ""
        return state
    
    failed_actions = state.get("failed_actions", [])
    actions_performed = state.get("actions_performed", [])
    client = OpenAI()

    # ---- Intent helper (scoped here) ----
    def infer_intent(goal: str) -> set[str]:
        g = (goal or "").lower()
        intent = set()
        if re.search(r'\b(setting|settings|preference|preferences|account|profile|billing|workspace)\b', g):
            intent.add("settings")
        if re.search(r'\b(theme|appearance|dark\s*mode|light\s*mode|notification|notifications)\b', g):
            intent.add("theme")
        if re.search(r'\b(search|find|filter|look\s*for)\b', g):
            intent.add("search")
        if re.search(r'\b(new|create|add|make)\b', g):
            intent.add("create")
        if re.search(r'\b(page|database|db|table|board|list)\b', g):
            intent.add("content")
        return intent

    intent = infer_intent(state.get("goal", ""))

    # Build terser string contexts (still kept for readability alongside snapshot)
    elements_list = [f"{i+1}. [{el['role']}] {el['name']}" for i, el in enumerate(visible_elements[:25])]
    elements_context = "\n".join(elements_list) if elements_list else "No elements found."
    failed_context = "\n⚠️ FAILED:\n" + "\n".join(failed_actions[-10:]) if failed_actions else ""
    actions_context = "\n✓ DONE:\n" + "\n".join(actions_performed[-10:]) if actions_performed else ""
    
    recent_action_summary = ""
    if actions_performed:
        last_action = actions_performed[-1]
        if "type:" in last_action:
            parts = last_action.split(":")
            if len(parts) >= 4:
                typed_text = parts[3]
                recent_action_summary = (
                    f"\n\n🎯 JUST COMPLETED: Typed '{typed_text}' into a textbox."
                    "\n   → Carefully check if this completes the user's goal before taking another action!"
                )
        elif "click:" in last_action and any(word in last_action for word in ["new", "add", "create"]):
            recent_action_summary = (
                "\n\n🎯 JUST COMPLETED: Clicked a creation button."
                "\n   → Look for newly available textboxes to type into (don't click creation buttons again!)"
            )
    just_recovered = any("recovery:escape" in act for act in actions_performed[-2:]) if actions_performed else False
    if just_recovered:
        actions_context += (
            "\n\n⚠️ JUST RECOVERED: Pressed Escape to close wrong modal/menu. "
            "Try HOVER to explore or click a DIFFERENT element!"
        )
    actions_context += recent_action_summary

    # 🔑 NEW: include compact JSON snapshot (cap length defensively)
    try:
        snapshot_json = json.dumps(snapshot, ensure_ascii=False)
    except Exception:
        snapshot_json = "{}"
    if len(snapshot_json) > 6000:
        # trim very large visible_elements/diff arrays to keep token usage sane
        mini = dict(snapshot)
        mini["visible_elements"] = snapshot.get("visible_elements", [])[:25]
        if "diff" in mini:
            mini["diff"] = {
                "added": snapshot.get("diff", {}).get("added", [])[:10],
                "removed": snapshot.get("diff", {}).get("removed", [])[:10],
            }
        snapshot_json = json.dumps(mini, ensure_ascii=False)

    system_message = f"""
You are an **autonomous UI navigation planner**.
Decide the *next concrete interaction* using ONLY the provided structured state and screenshot.

**State-grounding rules:**
- Treat the JSON snapshot as ground truth. Do not assume hidden elements exist.
- Prefer actions that operate on currently visible elements in the snapshot.
- Never open profile/workspace menus or Settings unless the goal explicitly mentions settings/preferences/account/billing/theme/appearance/notifications.

You will output ONE JSON object only:
{{
  "action_type": "click" | "type" | "keyboard" | "hover" | "scroll" | "noop",
  "role": "exact accessibility role",
  "name_pattern": "regex to match element name (case-insensitive)",
  "action_text": "text to type OR key name (Enter, Tab, Escape) OR scroll direction (down, up)"
}}

### STRATEGY
- Interpret the goal semantically (create≈add≈new≈+; search≈find≈filter; dark mode≈theme≈appearance).
- Navigation > Interaction > Confirmation.
- If you just clicked a creation button, do NOT click it again; type into the new textbox instead.
- If action buttons (Apply/Save/Edit) are missing, first select an item to reveal a details panel.
- If desired element isn’t visible, try HOVER or SCROLL before giving up.
- If the goal looks complete, return "noop".

### REGEX TIPS
- "log\\s*in", "dark|appearance", "new|add|create|\\+", "." (catch-all).
Avoid inline flags like (?i).

Return ONLY the JSON object.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Task (goal): {state["goal"]}

State Snapshot (JSON):
{snapshot_json}

Heuristics/history:
{failed_context}
{actions_context}

Visible elements (readable list):
{elements_context}

Return JSON only:"""
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{state['img_base64']}"}}
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
        
        goal_lower = (state.get("goal") or "").lower()
        login_el = next((el for el in visible_elements if "log" in el.get("name", "").lower() and "in" in el.get("name", "").lower()), None)
        if not login_el:
            login_el = next((el for el in visible_elements if "sign" in el.get("name", "").lower() and "in" in el.get("name", "").lower()), None)
        if login_el:
            print(f"🎯 Fallback: Login available - clicking to access workspace")
            state["action_type"] = "click"
            state["role"] = login_el['role']
            state["name_pattern"] = "log.*in|sign.*in"
            state["action_text"] = ""
            return state
        
        semantic_match = find_semantic_match(state["goal"], visible_elements)
        if semantic_match:
            el_name = semantic_match['name']
            state["action_type"] = "click"
            state["role"] = semantic_match['role']
            state["name_pattern"] = re.escape(el_name.split()[0]) if (el_name and ' ' in el_name) else re.escape(el_name) if el_name else ".*"
            state["action_text"] = ""
            return state
        
        textbox_el = next((el for el in visible_elements if el.get("role") == "textbox"), None)
        if textbox_el:
            import re as regex_module
            patterns = [
                r"following\s+in\s+it\s*:\s*(.+?)\.?$",
                r"in\s+it\s*:\s*(.+?)\.?$",
                r"write\s+the\s+following\s+in\s+it\s*:\s*(.+?)\.?$",
                r"following[:\s]+(.+?)(?:\s+in\s+it)?\.?$",
                r"write\s+(.+?)(?:\s+in\s+it)?\.?$",
                r"content[:\s]+(.+?)\.?$",
            ]
            text_to_type = None
            for pattern in patterns:
                m = regex_module.search(pattern, state["goal"] or "", regex_module.IGNORECASE)
                if m:
                    text_to_type = m.group(1).strip()
                    text_to_type = regex_module.sub(r'\s+in\s+(it|the\s+entry)\.?$', '', text_to_type, flags=regex_module.IGNORECASE)
                    break
            if not text_to_type:
                text_to_type = "New entry"
            state["action_type"] = "type"
            state["role"] = "textbox"
            state["name_pattern"] = ".*"
            state["action_text"] = text_to_type
            return state
        
        hover_explored = state.get("hover_explored", [])
        if len(visible_elements) < 15:
            hover_candidates = [el for el in visible_elements if el.get("role") in ["button", "link"]]
            for candidate in hover_candidates:
                hover_key = f"{candidate['role']}:{candidate['name']}"
                if hover_key not in hover_explored:
                    print(f"🔍 Fallback: UI is sparse - hovering to explore")
                    state["action_type"] = "hover"
                    state["role"] = candidate['role']
                    state["name_pattern"] = re.escape(candidate['name'])
                    state["action_text"] = ""
                    return state
        
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
        
        goal_lower = (state["goal"] or "").lower()

        # OVERRIDE -1: action button requires selecting an item first
        action_verbs = ["apply", "edit", "delete", "remove", "modify", "update", "save"]
        if action_type == "click" and any(verb in name_pattern.lower() for verb in action_verbs):
            item_links = [el for el in visible_elements if el.get("role") == "link" and len(el.get("name", "")) > 10]
            actions_performed_local = state.get("actions_performed", [])
            recent_clicks = [act for act in actions_performed_local[-3:] if "click:link" in act]
            if item_links and len(recent_clicks) == 0:
                print(f"⚠️  Looking for action button but haven't selected an item yet!")
                best_match = find_semantic_match(state["goal"], item_links)
                if best_match:
                    action_type = "click"
                    role = best_match['role']
                    name_pattern = re.escape(best_match['name'].split()[0]) if best_match['name'] else ".*"
                    action_text = ""

        # OVERRIDE 0: if goal implies typing/searching
        if action_type == "click" and role in ["combobox", "search", "searchbox", "textbox"]:
            if any(keyword in goal_lower for keyword in ["search", "find", "type", "enter", "write"]):
                search_keywords = ["search for", "find", "type", "enter", "write"]
                text_to_type = ""
                for keyword in search_keywords:
                    if keyword in goal_lower:
                        parts = goal_lower.split(keyword, 1)
                        if len(parts) > 1:
                            remaining = parts[1].strip()
                            text_to_type = remaining.split(" and ")[0].split(" then")[0].split(" in ")[0].strip()
                            break
                if text_to_type:
                    print(f"⚠️  GPT chose 'click' on search box but goal is to search - overriding to TYPE!")
                    action_type = "type"
                    action_text = text_to_type

        # OVERRIDE 1 (intent-gated settings/profile)
        if action_type == "noop":
            login_el = next(
                (el for el in visible_elements
                 if (("log" in el.get("name","").lower() and "in" in el.get("name","").lower())
                     or ("sign" in el.get("name","").lower() and "in" in el.get("name","").lower()))),
                None
            )
            if login_el:
                print("⚠️  noop returned but login is visible → clicking login to enter workspace")
                action_type = "click"
                role = login_el['role']
                name_pattern = "log.*in|sign.*in"
                action_text = ""
            else:
                # Check if we're ALREADY in Settings modal (tabs like "People", "General", "Preferences" visible)
                in_settings_modal = any(
                    el.get("role") == "tab" and any(kw in el.get("name", "").lower() for kw in ["people", "general", "preferences", "notifications", "connections"])
                    for el in visible_elements
                )
                
                if ("settings" in intent) or ("theme" in intent):
                    # Only click Settings if we're NOT already in Settings modal
                    if not in_settings_modal:
                        settings_keywords = ['setting', 'settings', 'preference', 'preferences', 'account', 'profile', 'workspace', 'appearance', 'theme', 'billing']
                        settings_el = next((el for el in visible_elements if any(k in el.get("name","").lower() for k in settings_keywords)), None)
                        if settings_el:
                            print(f"⚠️  noop → goal implies settings/theme; clicking '{settings_el['name']}'")
                        action_type = "click"
                        role = settings_el['role']
                        name_pattern = re.escape(settings_el['name'])
                        action_text = ""
                    else:
                        state["action_type"] = "noop"
                        state["role"] = ""
                        state["name_pattern"] = ""
                        state["action_text"] = ""
                        return state
                else:
                    state["action_type"] = "noop"
                    state["role"] = ""
                    state["name_pattern"] = ""
                    state["action_text"] = ""
                    return state

        # OVERRIDE 2: ensure TYPE has a matching textbox
        if action_type == "type":
            textbox_found = any(el.get("role") == "textbox" and re.search(name_pattern, el.get("name", ""), re.IGNORECASE) for el in visible_elements)
            if not textbox_found:
                any_textbox = next((el for el in visible_elements if el.get("role") == "textbox"), None)
                if any_textbox:
                    print(f"⚠️  Pattern '{name_pattern}' won't match any textbox → using '.*'")
                    name_pattern = ".*"

        # OVERRIDE 3: after creation, type the name/content
        just_created_something = any(
            "click:button" in act and any(kw in act for kw in ["new", "add", "create"])
            for act in actions_performed[-2:]
        )
        needs_to_type = any(keyword in goal_lower for keyword in ["write", "following", "name", "title", "call", "label", "add"])
        if action_type == "click" and just_created_something and needs_to_type:
            any_textbox = next((el for el in visible_elements if el.get("role") == "textbox"), None)
            typed_already = any("type:" in act for act in actions_performed)
            
            # CRITICAL: Don't trigger if template/creation options are still visible!
            # If we see buttons like "Empty database", "Empty page", "Tasks Tracker", etc.,
            # GPT should click those FIRST before typing.
            template_keywords = ["empty", "template", "tracker", "hub", "notes", "build with"]
            has_template_buttons = any(
                el.get("role") == "button" and any(kw in el.get("name", "").lower() for kw in template_keywords)
                for el in visible_elements
            )
            
            # Also check for search textbox (common in template pickers) - not the target!
            is_search_textbox = any_textbox and "search" in any_textbox.get("name", "").lower()
            
            if any_textbox and not typed_already and not has_template_buttons and not is_search_textbox:
                print(f"⚠️  Just created entry! Switching to TYPE mode")
                import re as regex_module
                patterns = [
                    r"name\s+(?:the\s+)?(?:task|entry|page|item|db|database)\s+as[:\s]+(.+?)\.?$",
                    r"name\s+it[:\s]+(.+?)\.?$",
                    r"title[:\s]+(.+?)\.?$",
                    r"call\s+it[:\s]+(.+?)\.?$",
                    r"write\s+the\s+following[:\s]+(.+?)\.?$",
                    r"write\s+(.+?)(?:\s+in\s+it)?\.?$",
                    r"content[:\s]+(.+?)\.?$",
                    r"add\s+(.+?)(?:\s+as\s+(?:a\s+)?(?:task|entry|item|database))?\.?$",
                ]
                text_to_type = None
                for pattern in patterns:
                    m = regex_module.search(pattern, state["goal"] or "", regex_module.IGNORECASE)
                    if m:
                        text_to_type = m.group(1).strip()
                        text_to_type = regex_module.sub(r'\s+in\s+(it|the\s+entry)\.?$', '', text_to_type, flags=regex_module.IGNORECASE)
                        text_to_type = regex_module.sub(r'\s+as\s+(a\s+)?(task|entry|item)\.?$', '', text_to_type, flags=regex_module.IGNORECASE)
                        break
                if not text_to_type:
                    quote_match = regex_module.search(r'["\'](.+?)["\']', state["goal"] or "")
                    text_to_type = quote_match.group(1) if quote_match else "New entry"
                
                # SMART TEXTBOX SELECTION: Prioritize NEW entry's textbox, not existing page
                all_textboxes = [el for el in visible_elements if el.get("role") == "textbox"]
                priority_keywords = ["untitled", "new page", "empty", "add title", "start typing"]
                
                # Look for a textbox with priority keywords (indicates NEW entry)
                priority_textbox = next(
                    (tb for tb in all_textboxes if any(kw in tb.get("name", "").lower() for kw in priority_keywords)),
                    None
                )
                
                # Use specific pattern to target the NEW entry's textbox
                if priority_textbox:
                    # Use first keyword match as pattern
                    matched_keyword = next(kw for kw in priority_keywords if kw in priority_textbox['name'].lower())
                    target_pattern = matched_keyword.replace(" ", "\\s+")  # e.g., "start\\s+typing"
                    print(f"  → Targeting NEW entry's textbox: '{priority_textbox['name']}'")
                else:
                    # Fallback: use wildcard (but warn about ambiguity)
                    target_pattern = ".*"
                    if len(all_textboxes) > 1:
                        print(f"  → Multiple textboxes found, using pattern '.*' (may be ambiguous)")
                
                action_type = "type"
                role = "textbox"
                name_pattern = target_pattern
                action_text = text_to_type

        # SAFETY CHECK: If GPT returned empty role/pattern for actions that need them, convert to noop
        if action_type in ["click", "type", "hover"] and (not role or not name_pattern):
            print(f"⚠️  GPT returned {action_type} with empty role/pattern - converting to noop")
            action_type = "noop"
            role = ""
            name_pattern = ""
            action_text = ""
        
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
        
        elif action_type == "scroll":
            # Scroll action: scroll the page to reveal more content
            direction = action_text.lower() if action_text else "down"
            print(f"📜 Scrolling {direction} to reveal more content...")
            
            if direction == "down":
                # Scroll down one page
                page.keyboard.press("PageDown")
            elif direction == "up":
                # Scroll up one page
                page.keyboard.press("PageUp")
            else:
                # Default to down
                page.keyboard.press("PageDown")
            
            page.wait_for_timeout(800)  # Wait for content to load
            print(f"✓ Scrolled {direction} successfully")
            
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
                
                # Try with better pattern if GPT suggested one
                element_found_with_pattern = False
                if better_pattern and better_pattern != name_pattern:
                    print(f"🔄 Retrying with new pattern: {better_pattern}")
                    try:
                        loc = page.get_by_role(role, name=re.compile(better_pattern, re.IGNORECASE)).first
                        loc.wait_for(state="visible", timeout=3000)
                        element_found_with_pattern = True
                    except:
                        print(f"⚠️  Still not found with new pattern")
                
                # If pattern didn't work, try alternative roles
                if not element_found_with_pattern:
                    print(f"⚠️  Trying alternative roles...")
                    
                    role_alternatives = {
                        "menuitem": ["option", "menuitemradio", "menuitemcheckbox", "combobox", "link", "button"],
                        "option": ["menuitem", "menuitemradio", "menuitemcheckbox", "combobox"],
                        "combobox": ["option", "menuitem", "button", "textbox", "search", "searchbox"],
                        "textbox": ["search", "searchbox", "combobox"],
                        "search": ["searchbox", "textbox", "combobox"],
                        "searchbox": ["search", "textbox", "combobox"],
                        "link": ["button", "menuitem", "option", "combobox"],
                        "button": ["link", "menuitem", "option", "combobox"]
                    }
                    
                    alternatives = role_alternatives.get(role, [])
                    element_found = False
                    
                    for alt_role in alternatives:
                        try:
                            print(f"🔄 Trying alternative role: {alt_role}")
                            alt_loc = page.get_by_role(alt_role, name=re.compile(name_pattern, re.IGNORECASE)).first
                            alt_loc.wait_for(state="visible", timeout=2000)
                            print(f"✓ Found with role '{alt_role}'!")
                            loc = alt_loc
                            role = alt_role  # Update role for later use
                            element_found = True
                            break
                        except:
                            continue
                    
                    if not element_found:
                        # Final fallback: Try CSS selectors for known patterns
                        print(f"⚠️  All role alternatives failed - trying CSS selector fallback...")
                        
                        # For action buttons, try common CSS patterns
                        if "apply" in name_pattern.lower():
                            css_selectors = [
                                f"button[aria-label*='{name_pattern}' i]",
                                f"button:has-text('{name_pattern}')",
                                "button[id*='apply']",
                                "button[class*='apply']",
                                "button.primary:has-text('Apply')"
                            ]
                            
                            for css_sel in css_selectors:
                                try:
                                    print(f"  🔍 Trying CSS: {css_sel}")
                                    css_loc = page.locator(css_sel).first
                                    css_loc.wait_for(state="visible", timeout=2000)
                                    print(f"  ✓ Found with CSS selector!")
                                    loc = css_loc
                                    element_found = True
                                    break
                                except:
                                    pass
                        
                        if not element_found:
                            raise wait_err  # Re-raise original error
            
            if action_type == "type":
                # Type action: fill input field
                print(f"📝 Typing '{action_text}' into [{role}]...")
                
                # FIRST: Check if there's a blocking modal - but DON'T close it if textbox is inside!
                try:
                    modals = page.locator("[role='dialog']:visible, [class*='modal']:visible, [class*='overlay']:visible").all()
                    if modals:
                        # Check if our target textbox is INSIDE any of these modals
                        textbox_inside_modal = False
                        for modal in modals:
                            try:
                                # Check if this modal contains a textbox matching our target
                                textboxes_in_modal = modal.locator(f"[role='{role}']:visible").all()
                                for tb in textboxes_in_modal:
                                    tb_name = tb.get_attribute("aria-label") or tb.get_attribute("placeholder") or ""
                                    if re.search(name_pattern, tb_name, re.IGNORECASE):
                                        textbox_inside_modal = True
                                        print(f"  ℹ️  Target textbox is INSIDE the modal - keeping it open")
                                        break
                            except:
                                pass
                            if textbox_inside_modal:
                                break
                        
                        # Only close modal if textbox is NOT inside it
                        if not textbox_inside_modal:
                            print("  ⚠️  Modal detected (not containing target) - closing it...")
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
                        loc.click(timeout=3000, force=True)  # Force click for complex nested DOMs
                        page.wait_for_timeout(500)  # Wait for any animations to complete
                        page.keyboard.press("Meta+A")  # Select all
                        page.keyboard.press("Backspace")  # Clear
                        page.wait_for_timeout(200)  # Let DOM process the clear
                        page.keyboard.type(action_text, delay=50)
                    else:
                        # Regular input/textarea
                        print("  (regular input field)")
                        loc.click(timeout=3000)
                        loc.fill(action_text)
                
                page.wait_for_timeout(500)
                print(f"✓ Typed '{action_text}' successfully")
                
                # Auto-press Enter for search boxes, comboboxes, or if goal mentions "search"
                goal_lower = state.get("goal", "").lower()
                is_search_context = (
                    role in ["combobox", "search", "searchbox"] or
                    "search" in goal_lower or
                    "find" in goal_lower
                )
                
                if is_search_context:
                    print(f"  ⌨️  Auto-pressing Enter to submit search...")
                    page.keyboard.press("Enter")
                    page.wait_for_timeout(1000)  # Wait for search results
                    print(f"  ✓ Enter pressed - search submitted!")
                
                # Remove cursor marker
                try:
                    page.evaluate("document.getElementById('cursor-marker-temp')?.remove()")
                except:
                    pass
                
                # Track successful action
                state.setdefault("actions_performed", []).append(action_key)
                
                # OPTIMIZATION 3: Check if typing this text completes the goal
                text_matches_goal = (
                    action_text.lower() in goal_lower or 
                    any(word in action_text.lower() for word in goal_lower.split() if len(word) > 3)
                )
                
                if text_matches_goal:
                    # Use GPT to determine if goal is actually complete
                    print(f"  🧠 Checking if typing '{action_text}' completes the goal...")
                    
                    goal_check_achieved = check_goal_achieved_by_state(
                        goal=state.get("goal", ""),
                        element_name=f"textbox",
                        current_state=f"Typed '{action_text}' into the textbox"
                    )
                    
                    if goal_check_achieved:
                        print(f"🎯 Goal '{state.get('goal')}' achieved after typing! Marking complete.")
                        state["goal_text_entered"] = True
                    else:
                        print(f"  ℹ️  Typed '{action_text}' but goal has additional steps - continuing...")
            
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
                    # Dropdown option or menu item (NOT combobox - those are for typing!)
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
                
                elif role == "combobox":
                    # Combobox: click to focus (user should TYPE into it on next step)
                    print(f"  📋 Clicking combobox to focus (ready for typing)...")
                    loc.click(timeout=5000)
                    page.wait_for_timeout(500)
                    print(f"✓ Combobox focused and ready for input!")
                
                else:
                    # Regular click (button, link, tab, etc.)
                    # Use force=True for complex nested DOMs (modern SPAs)
                    try:
                        loc.click(timeout=5000)
                    except:
                        # Retry with force if normal click fails
                        print("  ⚠️  Normal click failed - retrying with force=True...")
                        loc.click(timeout=5000, force=True)
                    
                    page.wait_for_timeout(1500)  # Wait for navigation/modal
                    print(f"✓ Click successful! Current URL: {page.url}")
                    
                    # Check if a modal opened
                    modal_visible = page.locator("[role='dialog']:visible, [role='alertdialog']:visible, [class*='modal']:visible").count() > 0
                    if modal_visible:
                        print("  ℹ️  Modal/dialog detected after click")
                    
                    # CHECK GOAL COMPLETION for action buttons (Send, Submit, Apply, Invite, Add, etc.)
                    # This is critical for workflows that don't involve typing
                    action_button_keywords = ["send", "submit", "apply", "invite", "add", "create", "save", "post", "publish"]
                    is_action_button = role == "button" and any(kw in name_pattern.lower() for kw in action_button_keywords)
                    
                    if is_action_button:
                        print(f"  🧠 Checking if clicking '{name_pattern}' completes the goal...")
                        
                        goal_achieved = check_goal_achieved_by_state(
                            goal=state.get("goal", ""),
                            element_name=f"button '{name_pattern}'",
                            current_state=f"Clicked button '{name_pattern}' successfully"
                        )
                        
                        if goal_achieved:
                            print(f"🎯 Goal '{state.get('goal')}' achieved after clicking button! Marking complete.")
                            state["goal_text_entered"] = True
                        else:
                            print(f"  ℹ️  Clicked button but goal may have additional steps - continuing...")
                
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
    url = input("Enter the website URL (e.g., https://example.com): ").strip()
    
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