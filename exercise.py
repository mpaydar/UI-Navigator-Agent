from playwright.sync_api import sync_playwright
import re
from openai import OpenAI
from dotenv import load_dotenv
import json
load_dotenv()

user_goal = input("Enter your goal: ")

# Get plan from GPT-4
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": """You are a UI automation planner for Playwright.
Generate a step-by-step plan to accomplish the goal using Playwright's role-based locators.

Return JSON with this format:
{
  "steps": [
    {
      "action": "click",
      "role": "link",
      "regex": "log\\s*in",
      "description": "Click the Log in link"
    },
    {
      "action": "type",
      "role": "textbox",
      "regex": "new\\s*page",
      "text": "My Journal Entry",
      "description": "Type in the page title field"
    }
  ]
}

CRITICAL:
- DO NOT use (?i) or other inline flags - they're not supported!
- Patterns will be matched case-insensitively automatically
- Use \\s* for flexible spacing (e.g., "log\\s*in" matches "Log in", "LOGIN", "log in")
- Use | for alternatives (e.g., "submit|continue" matches either)
- Available roles: link, button, textbox, heading, checkbox, switch
- Always start at https://www.notion.com/"""},
        {"role": "user", "content": user_goal}
    ]
)

plan = json.loads(response.choices[0].message.content)
print("\n📋 PLAN FROM GPT-4:")
print(json.dumps(plan, indent=2))

# Execute the plan with Playwright
with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    page = context.new_page()

    print("\n🌐 Navigating to Notion...")
    page.goto("https://www.notion.com/", wait_until="domcontentloaded", timeout=60000)
    page.wait_for_load_state("networkidle")

    # Execute each step
    for i, step in enumerate(plan.get("steps", []), 1):
        try:
            action = step.get("action", "click")
            role = step.get("role", "link")
            regex_pattern = step.get("regex", "")
            text = step.get("text", "")
            description = step.get("description", "")
            
            print(f"\n{'='*70}")
            print(f"STEP {i}: {description}")
            print(f"{'='*70}")
            print(f"Action: {action}")
            print(f"Role: {role}")
            print(f"Pattern: {regex_pattern}")
            
            # Find element using role-based locator
            print(f"🔍 Looking for [{role}] matching '{regex_pattern}'...")
            element = page.get_by_role(role, name=re.compile(regex_pattern, re.IGNORECASE)).first
            
            # Wait for it to be visible
            element.wait_for(state="visible", timeout=5000)
            
            if action == "type":
                print(f"📝 Typing: '{text}'")
                element.click()  # Focus first
                element.fill(text)
                page.wait_for_timeout(1000)
            elif action == "click":
                print(f"🖱️  Clicking...")
                element.click()
                page.wait_for_timeout(2000)  # Wait for navigation/changes
            
            print(f"✅ Step {i} completed!")
            
        except Exception as e:
            print(f"❌ Step {i} failed: {e}")
            print("Continuing anyway...")

    # Keep browser open to see result
    print("\n" + "="*70)
    print("🎉 ALL STEPS COMPLETED!")
    print("="*70)
    input("\nPress ENTER to close browser...")
    
    browser.close()
