# app.py
import os, re, io, json, base64, uuid, pathlib
from io import BytesIO
from PIL import Image
import gradio as gr

# --- Project roots & env defaults ---
ROOT = pathlib.Path(__file__).parent.resolve()
os.chdir(ROOT)
(pathlib.Path("data")).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CHAT_DB_PATH", str((ROOT / "data" / "chat_history.sqlite3").resolve()))
os.environ.setdefault("CSV_CATALOG_PATH", str((ROOT / "data" / "csv_clean" / "catalog.json").resolve()))

# --- Your agent & tools ---
from src.app.super_agent import build_super_agent
from src.databot.agent import get_history
from src.databot.tools import run_python_on_current_df  # must return a dict with image_b64 key when plotting

PROVIDER = os.getenv("PROVIDER", "ollama")
AGENT = build_super_agent(provider=PROVIDER)

PLOT_KEYWORDS = ("plot", "graph", "chart", "visualize", "bar chart", "line chart", "stacked", "heatmap")


def _sid(s):
    return s if isinstance(s, str) and s else f"ui-{uuid.uuid4().hex[:8]}"


def _extract_code_block(text: str) -> str | None:
    """Grab the first fenced python block, or any fenced code block if python not found."""
    if not isinstance(text, str):
        return None
    # ```python ... ```
    m = re.search(r"```python\s+(.+?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # generic ``` ... ```
    m = re.search(r"```\s+(.+?)```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


# app.py (only the helper and on_chat parts shown)

from PIL import Image
from io import BytesIO
import json, os

def _safe_json(obj) -> str:
    def _default(o):
        if isinstance(o, (bytes, bytearray)):
            return f"<bytes:{len(o)}>"
        return str(o)
    return json.dumps(obj, indent=2, default=_default)

def _maybe_get_image(obj_or_text):
    """
    Accept dict or JSON string. If it has `image_path`, open it as PIL.Image.
    Returns (chat_text, PIL.Image|None).
    """
    obj = obj_or_text
    if not isinstance(obj_or_text, dict):
        try:
            obj = json.loads(obj_or_text)
        except Exception:
            return str(obj_or_text), None

    if isinstance(obj, dict):
        # New path-based image
        if "image_path" in obj and obj["image_path"] and os.path.exists(obj["image_path"]):
            try:
                img = Image.open(obj["image_path"])
            except Exception:
                img = None
            clean = {k: v for k, v in obj.items() if k != "image_path"}
            # include the path in the chat text for traceability
            clean["image_path"] = obj["image_path"]
            return _safe_json(clean), img

        # No image_path; just show dict
        return _safe_json(obj), None

    return str(obj), None




def _looks_like_plot_request(text: str) -> bool:
    s = (text or "").lower()
    return any(k in s for k in PLOT_KEYWORDS)


def on_chat(message, chat_history, session_id):
    sid = _sid(session_id)
    history = get_history(sid)

    # 1) Ask the agent (it can call rag_query or select_csv / run_python_on_current_df, etc.)
    result = AGENT.invoke({"input": message, "chat_history": history.messages})
    agent_text = result.get("output", "")

    # 2) If it's a plot request, try to display the image:
    #    - first: tool-returned JSON with image_b64
    #    - second: code fence; we run it via run_python_on_current_df
    img = None
    display_text = agent_text

    if _looks_like_plot_request(message):
        # First try: maybe the agent already returned a JSON dict with image_b64 from our tool.
        txt, maybe_img = _maybe_get_image(agent_text)
        if maybe_img is not None:
            display_text, img = txt, maybe_img
        else:
            # Second try: did the agent return Python code? If so, run it through the tool.
            code = _extract_code_block(agent_text)
            if code:
                tool_out = run_python_on_current_df.invoke({"code": code})
                display_text, img = _maybe_get_image(tool_out)
            else:
                # No image & no code; just show its text response
                display_text = agent_text
    else:
        # Not a plot request -> just show the agent text
        display_text = agent_text

    # 3) Save chat memory
    history.add_user_message(message)
    history.add_ai_message(display_text)

    # 4) Update UI (gradio Chatbot type='messages')
    chat_history = list(chat_history or [])
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": display_text})

    return chat_history, img, sid, ""


with gr.Blocks(title="Finance RAG + Code Plot") as app:
    gr.Markdown("### 📈 Finance RAG + Code Plot\nAsk filing questions or say things like: _plot Apple iPhone vs Services (bar chart)_")
    sid = gr.State(value=f"ui-{uuid.uuid4().hex[:8]}")
    chat = gr.Chatbot(type="messages", height=520, label="Conversation")
    
    with gr.Row():
        box = gr.Textbox(placeholder="e.g., For AAPL latest 10‑Q MD&A revenue drivers; or Plot Apple iPhone vs Services (bar).", scale=4)
        send = gr.Button("Send", variant="primary", scale=1)
        clear = gr.Button("Clear", scale=1)
    img = gr.Image(label="Plot", height=360)
    send.click(on_chat, inputs=[box, chat, sid], outputs=[chat, img, sid, box])
    box.submit(on_chat, inputs=[box, chat, sid], outputs=[chat, img, sid, box])
    clear.click(lambda: ([], None, f"ui-{uuid.uuid4().hex[:8]}", ""), outputs=[chat, img, sid, box])

if __name__ == "__main__":
    app.launch(inbrowser=True)