# general_ui_agent_v3.py
# pip install langgraph langchain-openai selenium webdriver-manager python-dotenv
# .env: OPENAI_API_KEY=...

from typing import Annotated, Sequence, TypedDict, Literal, Optional, List, Tuple, Dict, Any
import os, json, time, pathlib, re, hashlib
from urllib.parse import urlparse
from dotenv import load_dotenv

# LangGraph / LangChain
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

# ===================== Setup & Globals =====================
load_dotenv()
STEP_DIR = pathlib.Path("steps"); STEP_DIR.mkdir(exist_ok=True)

def _truncate(s: str, n: int = 160) -> str:
    s = (s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return (s[: n-1] + "…") if len(s) > n else s

def _hash(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

def snap(tag: str) -> str:
    """Save a full-page screenshot of the current browser."""
    p = STEP_DIR / f"{int(time.time()*1000)}_{tag}.png"
    _SESSION.driver.save_screenshot(str(p))
    return str(p)

def _domain_allowed(allowlist: List[str], url: str) -> bool:
    if not allowlist: return True
    host = urlparse(url).netloc
    return any(host.endswith(urlparse(p).netloc) for p in allowlist)

# ================= Selenium Session (singleton) =================
class SeleniumSession:
    """Holds one Chrome session reused by tools."""
    driver: Optional[webdriver.Chrome] = None
    wait: Optional[WebDriverWait] = None
    allowlist: List[str] = []
    cookies_path: str = "session_cookies.json"

    def ensure(self, headed: bool = True, user_data_dir: Optional[str] = None, profile_dir: Optional[str] = None):
        if self.driver: return
        opts = webdriver.ChromeOptions()
        if headed:
            opts.add_argument("--start-maximized")
        else:
            opts.add_argument("--headless=new")
            opts.add_argument("--window-size=1366,860")
        if user_data_dir: opts.add_argument(f"--user-data-dir={user_data_dir}")
        if profile_dir: opts.add_argument(f"--profile-directory={profile_dir}")
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
        self.wait = WebDriverWait(self.driver, 30)

    def check_url(self, url: str):
        if not _domain_allowed(self.allowlist, url):
            raise PermissionError(f"URL '{url}' blocked by allowlist {self.allowlist}")

    def save_cookies(self):
        cookies = self.driver.get_cookies()
        pathlib.Path(self.cookies_path).write_text(json.dumps(cookies, indent=2))

    def load_cookies(self, domain_hint: str) -> bool:
        path = pathlib.Path(self.cookies_path)
        if not path.exists(): return False
        cookies = json.loads(path.read_text())
        self.driver.get(domain_hint)
        time.sleep(0.5)
        ok = 0
        for c in cookies:
            try:
                self.driver.add_cookie(c); ok += 1
            except Exception:
                pass
        self.driver.refresh()
        return ok > 0

_SESSION = SeleniumSession()

# ================ Helpers / Heuristics ==================
def _detect_logged_in(url: str, dom_html: str) -> bool:
    """
    Heuristic: True if app likely past login.
    Works across many apps, not Linear-specific.
    """
    if not url:
        return False

    # Common post-login signals
    path_hits = re.search(r"/(team|issues?|projects?|tasks?|workspace|dashboard|inbox|active|backlog)(/|$)", url, re.I)
    host_hits = re.search(r"(app\.)", url)  # app subdomain is common
    if path_hits or host_hits:
        return True

    # DOM fallback (common nav labels)
    if dom_html and re.search(r"\b(Inbox|My issues|Projects|Backlog|Active|Dashboard|Home)\b", dom_html, re.I):
        return True

    # Not obviously login page
    if "login" not in url and "signin" not in url and "auth" not in url:
        return True

    return False

def _by_from_locator(kind: str, value: str, name: Optional[str]=None) -> Tuple[By, str]:
    """Translate abstract locators to Selenium selectors (general, not app specific)."""
    if kind == "role":
        if value == "link":
            if name: return By.XPATH, f'//a[normalize-space(.)={json.dumps(name)}]'
            return By.TAG_NAME, "a"
        if value == "button":
            if name: return By.XPATH, f'//*[(self::button or @role="button") and normalize-space(.)={json.dumps(name)}]'
            return By.XPATH, '//*[self::button or @role="button"]'
        if value == "textbox":
            if not name: return By.XPATH, '//*[self::input or self::textarea]'
            nm = name; nml = name.lower()
            xp = (
                f'//input[@aria-label={json.dumps(nm)} or @placeholder={json.dumps(nm)}'
                f' or translate(@name,"ABCDEFGHIJKLMNOPQRSTUVWXYZ","abcdefghijklmnopqrstuvwxyz")={json.dumps(nml)}]'
                f' | //textarea[@aria-label={json.dumps(nm)} or @placeholder={json.dumps(nm)}]'
                f' | //label[normalize-space(.)={json.dumps(nm)}]/following::input[1]'
            )
            return By.XPATH, xp
        if name:
            return By.XPATH, f'//*[@role="{value}" and normalize-space(.)={json.dumps(name)}]'
        return By.CSS_SELECTOR, f'[role="{value}"]'
    if kind == "testid":
        return By.CSS_SELECTOR, f'[data-testid="{value}"]'
    if kind == "text":
        return By.XPATH, f'//*[normalize-space(.)={json.dumps(value)}]'
    if kind == "css":
        return By.CSS_SELECTOR, value
    if kind == "xpath":
        return By.XPATH, value
    raise ValueError(f"Unknown locator kind: {kind}")

def _is_displayed(el) -> bool:
    try:
        return el.is_displayed() and el.is_enabled()
    except Exception:
        return False

def _highlight(el):
    try:
        _SESSION.driver.execute_script("arguments[0].style.outline='3px solid magenta';", el)
    except Exception:
        pass

def _candidate_dialog_roots() -> List[Any]:
    """Return likely modal/dialog containers first, then document."""
    d = _SESSION.driver
    candidates = []
    selectors = [
        # very common dialog patterns
        "[role='dialog']",
        "[aria-modal='true']",
        ".ReactModal__Content",
        ".modal, .Modal, .ant-modal-content, .mantine-Modal-content, .MuiDialog-container, .chakra-modal__content",
        "[data-overlay-container], [data-state='open']",
    ]
    for sel in selectors:
        try:
            for el in d.find_elements(By.CSS_SELECTOR, sel):
                if _is_displayed(el):
                    candidates.append(el)
        except Exception:
            pass
    candidates.append(d.find_element(By.TAG_NAME, "body"))
    return candidates

def _visible_elements(root, by: By, selector: str) -> List[Any]:
    try:
        els = root.find_elements(by, selector)
    except Exception:
        return []
    return [e for e in els if _is_displayed(e)]

def _smart_text_fields(scope: str = "dialog") -> List[Any]:
    """Find visible text-like fields (input/textarea/contenteditable), dialog-first."""
    roots = _candidate_dialog_roots() if scope == "dialog" else [_SESSION.driver.find_element(By.TAG_NAME, "body")]
    for r in roots:
        # Inputs / textareas
        candidates: List[Any] = []
        for sel in [
            "input[type='text']",
            "input:not([type]), input[type='search'], input[type='url'], input[type='email']",
            "textarea",
            "[contenteditable=''], [contenteditable='true']",
            "input[placeholder], textarea[placeholder]",
        ]:
            candidates += _visible_elements(r, By.CSS_SELECTOR, sel)
        # Filter out obvious non-editables (checkbox, radio etc.)
        res = []
        for e in candidates:
            try:
                tag = e.tag_name.lower()
                typ = (e.get_attribute("type") or "").lower()
                if tag == "input" and typ in {"checkbox","radio","hidden","file","button","submit"}:
                    continue
                res.append(e)
            except Exception:
                continue
        if res:
            return res
    return []

def _smart_buttons(texts: List[str], scope: str = "dialog") -> List[Any]:
    """Find visible buttons whose text contains any candidate string (case-insensitive)."""
    roots = _candidate_dialog_roots() if scope == "dialog" else [_SESSION.driver.find_element(By.TAG_NAME, "body")]
    pat = re.compile("|".join([re.escape(t) for t in texts]), re.I) if texts else re.compile(r".+")
    for r in roots:
        els: List[Any] = []
        # buttons, role=button, inputs that act as buttons
        for sel in ["button", "[role='button']", "input[type='submit']", "input[type='button']"]:
            els += _visible_elements(r, By.CSS_SELECTOR, sel)
        # Now score by text match
        scored = []
        for e in els:
            try:
                txt = (e.text or "").strip()
                if not txt:
                    txt = (e.get_attribute("value") or "").strip()
                if pat.search(txt):
                    scored.append((len(txt), e))
            except Exception:
                continue
        if scored:
            # Prefer shorter/more exact-looking labels
            scored.sort(key=lambda t: t[0])
            return [e for _, e in scored]
    return []

# ===================== LangGraph State =======================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ==================== Primitive Action Tools ==================
@tool
def start_session(allowlist: List[str],
                  headed: bool = True,
                  user_data_dir: Optional[str] = None,
                  profile_dir: Optional[str] = None) -> str:
    """Start browser session with domain allowlist. Headed mode shows browser window."""
    _SESSION.allowlist = allowlist
    _SESSION.ensure(headed=headed, user_data_dir=user_data_dir, profile_dir=profile_dir)
    return json.dumps({"ok": True, "headed": headed, "allowlist": allowlist})

@tool
def open_url(url: str) -> str:
    """Navigate to URL (must be allowlisted) and screenshot."""
    _SESSION.ensure()
    _SESSION.check_url(url)
    _SESSION.driver.get(url)
    _SESSION.wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
    shot = snap("after_nav")
    return json.dumps({"ok": True, "url": _SESSION.driver.current_url, "screenshot": shot})

@tool
def restore_cookies(domain_hint: str) -> str:
    """Load saved cookies from disk to restore login session."""
    _SESSION.ensure()
    ok = _SESSION.load_cookies(domain_hint)
    return json.dumps({"ok": ok, "url": _SESSION.driver.current_url})

@tool
def persist_cookies() -> str:
    """Save current cookies to disk for future sessions."""
    _SESSION.ensure()
    _SESSION.save_cookies()
    return json.dumps({"ok": True})

@tool
def wait_url_contains(fragment: str, timeout_sec: int = 30) -> str:
    """Wait until current URL contains the specified fragment."""
    _SESSION.ensure()
    WebDriverWait(_SESSION.driver, timeout_sec).until(lambda d: fragment in (d.current_url or ""))
    return json.dumps({"ok": True, "url": _SESSION.driver.current_url})

@tool
def click(kind: Literal["role","testid","text","css","xpath"], value: str, name: Optional[str]=None) -> str:
    """Click an element by locator. Takes before/after screenshots."""
    _SESSION.ensure()
    by, sel = _by_from_locator(kind, value, name)
    el = _SESSION.wait.until(EC.element_to_be_clickable((by, sel)))
    _highlight(el)
    before = snap("before_click")
    ActionChains(_SESSION.driver).move_to_element(el).pause(0.12).click(el).perform()
    time.sleep(0.25)
    after = snap("after_click")
    return json.dumps({"ok": True, "before": before, "after": after})

@tool
def fill(kind: Literal["role","testid","text","css","xpath"], value: str, text: str, name: Optional[str]=None) -> str:
    """Fill a specific located input field with text. Takes before/after screenshots."""
    _SESSION.ensure()
    by, sel = _by_from_locator(kind, value, name)
    el = _SESSION.wait.until(EC.visibility_of_element_located((by, sel)))
    _highlight(el)
    before = snap("before_fill")
    try: el.clear()
    except Exception: pass
    el.click()
    # Select-all then type to be robust
    if os.uname().sysname == "Darwin":
        el.send_keys(Keys.COMMAND, "a")
    else:
        el.send_keys(Keys.CONTROL, "a")
    el.send_keys(text)
    # JS fallback for frameworks that ignore send_keys changes until blur
    try:
        _SESSION.driver.execute_script("""
            const el = arguments[0], val = arguments[1];
            if (el.isContentEditable) { el.textContent = val; }
            else if ('value' in el) { el.value = val; }
            el.dispatchEvent(new Event('input', {bubbles:true}));
            el.dispatchEvent(new Event('change', {bubbles:true}));
        """, el, text)
    except Exception:
        pass
    time.sleep(0.15)
    after = snap("after_fill")
    return json.dumps({"ok": True, "before": before, "after": after, "text_len": len(text)})

@tool
def screenshot(scope: Literal["page","element"]="page",
               kind: Optional[Literal["role","testid","text","css","xpath"]]=None,
               value: Optional[str]=None,
               name: Optional[str]=None) -> str:
    """Take a screenshot of the full page or specific element."""
    _SESSION.ensure()
    if scope == "page" or not kind or not value:
        path = snap("page")
        return json.dumps({"ok": True, "path": path})
    by, sel = _by_from_locator(kind, value, name)
    el = _SESSION.wait.until(EC.visibility_of_element_located((by, sel)))
    _highlight(el)
    path = snap(f"elem_{kind}")
    return json.dumps({"ok": True, "path": path})

# ==================== Generic High-Value Tools ====================
@tool
def fill_first_text_field(text: str, scope: Literal["dialog","page"] = "dialog") -> str:
    """
    Fill the first visible text field in the current scope (dialog-first by default).
    Supports input, textarea, and contenteditable nodes. JS fallback included.
    """
    _SESSION.ensure()
    fields = _smart_text_fields(scope=scope)
    if not fields:
        shot = snap("fill_first_text_field_no_field")
        return json.dumps({"ok": False, "error": "no text fields found", "screenshot": shot})
    el = fields[0]
    _SESSION.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
    _highlight(el)
    before = snap("before_fill_any")
    try:
        el.click()
        # Select-all then type
        if os.uname().sysname == "Darwin":
            el.send_keys(Keys.COMMAND, "a")
        else:
            el.send_keys(Keys.CONTROL, "a")
        el.send_keys(text)
        # JS fallback to ensure frameworks pick it up
        _SESSION.driver.execute_script("""
            const el = arguments[0], val = arguments[1];
            if (el.isContentEditable) { el.textContent = val; }
            else if ('value' in el) { el.value = val; }
            el.dispatchEvent(new Event('input', {bubbles:true}));
            el.dispatchEvent(new Event('change', {bubbles:true}));
        """, el, text)
    except Exception as e:
        shot = snap("fill_first_text_field_error")
        return json.dumps({"ok": False, "error": str(e), "screenshot": shot})
    after = snap("after_fill_any")
    return json.dumps({"ok": True, "before": before, "after": after})

@tool
def click_button_by_text(candidates: List[str], scope: Literal["dialog","page"] = "dialog") -> str:
    """
    Click a visible button whose text contains any of the candidate strings (case-insensitive).
    Works with <button>, [role=button], and input[type=submit/button].
    """
    _SESSION.ensure()
    btns = _smart_buttons(candidates, scope=scope)
    if not btns:
        shot = snap("click_button_by_text_no_button")
        return json.dumps({"ok": False, "error": "no matching buttons", "screenshot": shot})
    el = btns[0]
    _SESSION.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
    _highlight(el)
    before = snap("before_click_button")
    try:
        ActionChains(_SESSION.driver).move_to_element(el).pause(0.1).click(el).perform()
    except Exception as e:
        shot = snap("click_button_by_text_error")
        return json.dumps({"ok": False, "error": str(e), "screenshot": shot})
    time.sleep(0.3)
    after = snap("after_click_button")
    return json.dumps({"ok": True, "before": before, "after": after})

# ==================== Snapshot/Digest Tool ====================
def _attr(e, name) -> str:
    try: return e.get_attribute(name) or ""
    except Exception: return ""

def _collect_digest(limit_each: int = 70) -> Dict[str, Any]:
    d = _SESSION.driver
    dom_html = d.execute_script("return document.documentElement.outerHTML || '';")

    digest: Dict[str, Any] = {
        "url": d.current_url or "",
        "title": d.title or "",
        "dom_hash": _hash(dom_html)[:16],  # progress detection
        "logged_in_guess": _detect_logged_in(d.current_url, dom_html),
        "headings": [],
        "links": [],
        "buttons": [],
        "inputs": [],
    }
    # headings
    for tag in ["h1","h2","h3"]:
        try:
            for h in d.find_elements(By.TAG_NAME, tag)[:limit_each]:
                t = _truncate(h.text or "")
                if t: digest["headings"].append({"tag": tag, "text": t})
        except Exception: pass
    # links
    try:
        for a in d.find_elements(By.TAG_NAME, "a")[:limit_each]:
            t = _truncate(a.text or ""); href = _attr(a, "href")
            if t or href:
                digest["links"].append({"text": t, "href": href, "role": _attr(a,"role")})
    except Exception: pass
    # buttons
    try:
        for b in d.find_elements(By.TAG_NAME, "button")[:limit_each]:
            t = _truncate(b.text or "")
            digest["buttons"].append({"text": t, "role": _attr(b,"role")})
    except Exception: pass
    # inputs / textareas
    try:
        ins = d.find_elements(By.TAG_NAME, "input")[:limit_each]
        tas = d.find_elements(By.TAG_NAME, "textarea")[:limit_each]
        for e in ins + tas:
            digest["inputs"].append({
                "tag": e.tag_name,
                "type": _attr(e,"type"),
                "name": _attr(e,"name"),
                "id": _attr(e,"id"),
                "placeholder": _truncate(_attr(e,"placeholder")),
                "aria-label": _truncate(_attr(e,"aria-label")),
                "data-testid": _attr(e,"data-testid"),
            })
    except Exception: pass
    # Size guard
    as_json = json.dumps(digest)
    if len(as_json) > 180_000:
        digest["links"] = digest["links"][:40]
        digest["buttons"] = digest["buttons"][:40]
        digest["inputs"] = digest["inputs"][:40]
    return digest

@tool
def snapshot_elements() -> str:
    """Collect digest of interactive elements + screenshot."""
    _SESSION.ensure()
    dig = _collect_digest()
    img = snap("snapshot")
    return json.dumps({"ok": True, "digest": dig, "screenshot": img})

# =================== Planner & Parsing ==================
PLANNER_SYS = SystemMessage(content=(
    "You are a strict, general UI automation planner.\n"
    "INPUT: {\"goal\": str, \"page\": digest, \"already_logged_in\": bool, \"credentials\": {\"email\": str, \"password\": str}}\n"
    "OUTPUT: ONLY valid JSON with schema:\n"
    '{\"steps\":[{\"op\":\"click\"|\"fill\"|\"fill_first_text_field\"|\"click_button_by_text\"|\"screenshot\"|\"open_url\"|\"wait_url_contains\",'
    '\"locator\":{\"strategy\":\"role\"|\"testid\"|\"text\"|\"css\"|\"xpath\",\"value\":str,\"name\":str|null},'
    '\"text\":str|null,\"candidates\": [str]|null, \"scope\": \"dialog\"|\"page\"|null}],'
    '\"rationale\":str,\"confidence\":number}\n\n'
    "Rules:\n"
    "- If already_logged_in is true, SKIP login steps and focus ONLY on the goal.\n"
    "- For creation modals: use fill_first_text_field(name, scope='dialog') then click_button_by_text(['Create project','Create','Save'], scope='dialog').\n"
    "- Prefer dialog scope when a modal is present.\n"
    "- Keep steps atomic; end with a screenshot.\n"
))

def _extract_json_from_text(text: str) -> Optional[str]:
    text = (text or "").strip()
    if text.startswith("```json"): text = text[7:]
    if text.startswith("```"): text = text[3:]
    if text.endswith("```"): text = text[:-3]
    text = text.strip()
    start = text.find("{"); end = text.rfind("}")
    if start >= 0 and end > start: return text[start:end+1]
    return text

def _coerce_plan_from_text(text: str) -> Dict[str, Any]:
    if not text:
        return {"steps": [{"op": "screenshot"}], "rationale": "empty", "confidence": 0.0}
    js = _extract_json_from_text(text)
    try:
        obj = json.loads(js)
        if isinstance(obj, dict) and "steps" in obj: return obj
    except Exception:
        pass
    return {"steps": [{"op": "screenshot"}], "rationale": "could not parse", "confidence": 0.0}

@tool
def propose_actions(goal: str, digest: Optional[Dict[str, Any]] = None) -> str:
    """Generate action plan to achieve goal based on current page digest."""
    if digest is None:
        digest = _collect_digest()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    planner_input = {
        "goal": goal,
        "page": digest,
        "already_logged_in": digest.get("logged_in_guess", False),
        "credentials": {
            "email": os.getenv("LINEAR_EMAIL", ""),
            "password": os.getenv("LINEAR_PASSWORD", "")
        }
    }
    resp = llm.invoke([PLANNER_SYS, HumanMessage(content=json.dumps(planner_input, ensure_ascii=False))])
    content = resp.content
    plan = content if isinstance(content, dict) else _coerce_plan_from_text(str(content))
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        plan = {"steps": [{"op": "screenshot"}], "rationale": "invalid plan; fallback", "confidence": 0.0}
    return json.dumps({"ok": True, "plan": plan})

# ================= Execute plan with verification =================
def _resolve_locator(loc: Dict[str, Any]) -> Tuple[By, str]:
    strat = loc.get("strategy"); val = loc.get("value"); nm = loc.get("name")
    if strat == "placeholder":  # convenience
        strat = "css"; val = f"input[placeholder*='{val}'], textarea[placeholder*='{val}']"
    if strat not in {"role","testid","text","css","xpath"}: raise ValueError(f"Unsupported strategy: {strat}")
    if not val: raise ValueError("Locator missing 'value'")
    return _by_from_locator(strat, val, nm)

@tool
def execute_plan(plan: Dict[str, Any], allow_screens: bool = True) -> str:
    """Execute a sequence of UI actions from the plan with verification screenshots."""
    _SESSION.ensure()
    out_steps = []
    for idx, st in enumerate(plan.get("steps", []), start=1):
        op = st.get("op")
        try:
            if op == "open_url":
                url = st.get("text") or st.get("url") or st.get("locator", {}).get("value") or ""
                if not url: raise ValueError("open_url requires URL")
                _SESSION.check_url(url)
                _SESSION.driver.get(url)
                _SESSION.wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
                path = snap(f"{idx:02d}_after_nav")
                out_steps.append({"op": op, "ok": True, "shot": path, "url": _SESSION.driver.current_url})

            elif op == "wait_url_contains":
                frag = st.get("text") or ""
                if not frag: raise ValueError("wait_url_contains requires 'text'")
                WebDriverWait(_SESSION.driver, 30).until(lambda d: frag in (d.current_url or ""))
                out_steps.append({"op": op, "ok": True, "url": _SESSION.driver.current_url})

            elif op == "fill_first_text_field":
                scope = st.get("scope") or "dialog"
                res = json.loads(fill_first_text_field.invoke({"text": st.get("text") or "", "scope": scope}))
                out_steps.append({"op": op, **res})

            elif op == "click_button_by_text":
                scope = st.get("scope") or "dialog"
                cands = st.get("candidates") or []
                res = json.loads(click_button_by_text.invoke({"candidates": cands, "scope": scope}))
                out_steps.append({"op": op, **res})

            elif op == "click":
                by, sel = _resolve_locator(st.get("locator", {}))
                el = _SESSION.wait.until(EC.element_to_be_clickable((by, sel)))
                _highlight(el)
                before = snap(f"{idx:02d}_before_click")
                ActionChains(_SESSION.driver).move_to_element(el).pause(0.12).click(el).perform()
                time.sleep(0.25)
                after = snap(f"{idx:02d}_after_click") if allow_screens else ""
                out_steps.append({"op": op, "ok": True, "before": before, "after": after})

            elif op == "fill":
                by, sel = _resolve_locator(st.get("locator", {}))
                text = st.get("text") or ""
                el = _SESSION.wait.until(EC.visibility_of_element_located((by, sel)))
                _highlight(el)
                before = snap(f"{idx:02d}_before_fill")
                try: el.clear()
                except Exception: pass
                el.click()
                if os.uname().sysname == "Darwin":
                    el.send_keys(Keys.COMMAND, "a")
                else:
                    el.send_keys(Keys.CONTROL, "a")
                el.send_keys(text)
                try:
                    _SESSION.driver.execute_script("""
                        const el = arguments[0], val = arguments[1];
                        if (el.isContentEditable) { el.textContent = val; }
                        else if ('value' in el) { el.value = val; }
                        el.dispatchEvent(new Event('input', {bubbles:true}));
                        el.dispatchEvent(new Event('change', {bubbles:true}));
                    """, el, text)
                except Exception:
                    pass
                after = snap(f"{idx:02d}_after_fill") if allow_screens else ""
                out_steps.append({"op": op, "ok": True, "before": before, "after": after, "len": len(text)})

            elif op == "screenshot":
                p = snap(f"{idx:02d}_page")
                out_steps.append({"op": op, "ok": True, "path": p})

            else:
                out_steps.append({"op": op, "ok": False, "error": f"unsupported op '{op}'"})
        except Exception as e:
            fail_shot = snap(f"{idx:02d}_error")
            out_steps.append({"op": op, "ok": False, "error": str(e), "shot": fail_shot})
            break

    return json.dumps({"ok": True, "executed": out_steps})

# ================= Iterative high-level loop =================
@tool
def run_goal(goal: str, max_iters: int = 12, wait_hint: Optional[str] = None) -> str:
    """
    Iteratively: snapshot -> propose -> execute -> verify progress.
    Stops when no progress (same url+dom_hash) or plan is only screenshots.
    """
    _SESSION.ensure()
    history = []
    prev_url, prev_hash = "", ""
    no_progress_count = 0

    for i in range(1, max_iters+1):
        dig = _collect_digest()
        cur_url, cur_hash = dig["url"], dig["dom_hash"]
        shot0 = snap(f"iter{i:02d}_snapshot")

        if i > 1 and cur_url == prev_url and cur_hash == prev_hash:
            no_progress_count += 1
            max_no_progress = 5 if dig.get("logged_in_guess", False) else 2
            if no_progress_count >= max_no_progress:
                history.append({"iter": i, "status": "stuck_no_progress", "url": cur_url})
                break
        else:
            no_progress_count = 0

        # Plan
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        planner_input = {
            "goal": goal,
            "page": dig,
            "already_logged_in": dig.get("logged_in_guess", False),
            "credentials": {"email": os.getenv("LINEAR_EMAIL",""), "password": os.getenv("LINEAR_PASSWORD","")}
        }
        resp = llm.invoke([PLANNER_SYS, HumanMessage(content=json.dumps(planner_input, ensure_ascii=False))])
        plan = resp.content if isinstance(resp.content, dict) else _coerce_plan_from_text(str(resp.content))
        steps = plan.get("steps", [])
        print(f"📋 Iter {i}: steps -> {[s.get('op') for s in steps]}")

        if not steps or all(s.get("op") == "screenshot" for s in steps):
            history.append({"iter": i, "status": "no_actionable_steps", "url": cur_url})
            break

        # Execute via the tool (reuses robust code paths + screenshots)
        exec_res = json.loads(execute_plan.invoke({"plan": plan}))
        history.append({"iter": i, "plan": plan, "executed": exec_res})

        if wait_hint:
            try:
                WebDriverWait(_SESSION.driver, 5).until(lambda d: wait_hint in (d.current_url or ""))
            except Exception:
                pass

        dig2 = _collect_digest()
        prev_url, prev_hash = dig2["url"], dig2["dom_hash"]

    final_shot = snap("final")
    final_url = _SESSION.driver.current_url
    final_dom = _SESSION.driver.execute_script("return document.documentElement.outerHTML || '';")
    success = _detect_logged_in(final_url, final_dom)

    return json.dumps({
        "ok": True,
        "history": history,
        "final_url": final_url,
        "final_screenshot": final_shot,
        "logged_in": success
    })

# ================= Wire tools into an agent graph =================
TOOLS = [
    start_session, open_url, restore_cookies, persist_cookies,
    wait_url_contains, click, fill, screenshot,
    fill_first_text_field, click_button_by_text,
    snapshot_elements, propose_actions, execute_plan, run_goal
]

llm_router = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(TOOLS)

ROUTER_SYS = SystemMessage(content=(
    "Coordinator. Use tools ONLY. Typical cycle:\n"
    "start_session -> (restore_cookies?) -> open_url -> run_goal.\n"
    "Do not emit plain text. Keep going until run_goal completes."
))

def agent_node(state: AgentState) -> AgentState:
    resp = llm_router.invoke([ROUTER_SYS] + state["messages"])
    return {"messages": [resp]}

def should_continue(state: AgentState):
    last = state["messages"][-1]
    return "continue" if getattr(last, "tool_calls", None) else "end"

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
graph.add_edge("tools", "agent")
app = graph.compile()

# ============================ Demo ============================
if __name__ == "__main__":
    allow = [
        os.getenv("ALLOW1", "https://linear.app"),
        os.getenv("ALLOW2", "https://app.linear.app"),
    ]
    seed_url = os.getenv("SEED_URL", "https://linear.app/login")

    goal = os.getenv(
        "GOAL",
        "Create a new project named 'testbayat' and save it, then take a final screenshot."
    )

    inputs = {"messages": [
        HumanMessage(content=json.dumps({
            "goal": goal,
            "allowlist": allow,
            "seed_url": seed_url,
            "plan_hint": [
                {"tool":"start_session","args":{
                    "allowlist": allow,
                    "headed": True
                }},
                {"tool":"restore_cookies","args":{"domain_hint": allow[0]}},
                {"tool":"open_url","args":{"url": seed_url}},
                # The router continues with run_goal which uses propose_actions -> execute_plan
                {"tool":"run_goal","args":{"goal": goal, "max_iters": 12, "wait_hint":"app."}}
            ]
        }))
    ]}

    for s in app.stream(inputs, stream_mode="values", config={"recursion_limit": 30}):
        msg = s["messages"][-1]
        try:
            msg.pretty_print()
        except Exception:
            print(msg)

    print("\n" + "="*80)
    print("✅ Automation complete! Browser will stay open for 5 minutes.")
    print(f"📸 Screenshots saved to: {STEP_DIR.absolute()}")
    print("="*80)
    print("\nPress Ctrl+C to close browser now, or it will auto-close in 5 minutes...")
    try:
        time.sleep(300)
    except KeyboardInterrupt:
        print("\n👋 Closing browser...")
    if _SESSION.driver:
        _SESSION.driver.quit()
        print("Browser closed.")
