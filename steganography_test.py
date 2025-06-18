from __future__ import annotations

import asyncio
import json
import re
from collections import namedtuple
from typing import Optional

from openai import OpenAI

from config import (
    OPENROUTER_API_KEY,
    MODELS,
    NUM_ATTEMPTS,
    NUM_BRANCHES,
    MAX_TOKENS,
)

MAX_CONCURRENCY = 5
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

# ───────────────────────── prompt helpers ────────────────────────────
def load_prompt(fname: str) -> str | None:
    try:
        with open(fname, encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"❌ Could not find {fname}")
        return None


STEGANOGRAPHY_PROMPT = load_prompt("steganography_prompt.txt")
SYSTEM_PROMPT = load_prompt("system_prompt.txt")

# ───────────────────── sync→async OpenAI wrapper ─────────────────────
def _chat_sync(model: str, msgs: list[dict[str, str]]) -> str | None:
    try:
        resp = client.chat.completions.create(
            model=model, messages=msgs, max_tokens=MAX_TOKENS, temperature=0.7
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"<<API error: {e}>>"


async def chat(model: str, msgs: list[dict[str, str]], sem: asyncio.Semaphore) -> str | None:
    async with sem:
        return await asyncio.to_thread(_chat_sync, model, msgs)

# ─────────────────────────── parsing helpers ─────────────────────────
Parsed = namedtuple("Parsed", ["reasoning", "answer"])
_TAG_RE = re.compile(
    r"<(?P<tag>answer|reasoning)>\s*(?P<body>.*?)(?:</\1>|$)",
    re.IGNORECASE | re.DOTALL,
)

def _strip_tags(txt: str) -> str:
    return re.sub(r"</?\s*(answer|reasoning)\s*>", "", txt, flags=re.IGNORECASE).strip()

def separate_reasoning_and_answer(txt: str | None) -> Parsed:
    if not txt:
        return Parsed("(no reasoning)", "")
    tagged = {m.group("tag").lower(): m.group("body").strip() for m in _TAG_RE.finditer(txt)}
    ans, reas = tagged.get("answer"), tagged.get("reasoning")
    if ans is None and reas is None:
        lines = [l for l in (ln.strip() for ln in txt.splitlines()) if l]
        return Parsed("\n".join(lines[:-1]).strip() or "(no reasoning)", lines[-1] if lines else "")
    if ans is None:
        ans = _strip_tags(_TAG_RE.sub("", txt)) or "<empty answer>"
    if reas is None:
        reas = _strip_tags(_TAG_RE.sub("", txt)) or "(no reasoning)"
    return Parsed(reas, ans)

def extract_yes_no(txt: str | None) -> str | None:
    if not txt:
        return None
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", txt, re.IGNORECASE | re.DOTALL)
    block = m.group(1) if m else txt
    m2 = re.search(r"\b(yes|no)\b", block, re.IGNORECASE)
    return m2.group(1).lower() if m2 else None

def determine_claimed_color(question: str, answer: str | None) -> str | None:
    if question == "is it white":
        return "white" if answer == "yes" else "black"
    if question == "is it black":
        return "black" if answer == "yes" else "white"
    return None

# ───────────────────── attempt-nivå hjälp­funktioner ─────────────────
AttemptRoot = namedtuple("AttemptRoot", ["msgs", "answer0"])

async def get_root_reply(model: str, idx: int, sem: asyncio.Semaphore) -> AttemptRoot | None:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": STEGANOGRAPHY_PROMPT},
    ]
    resp = await chat(model, msgs, sem)
    if not resp:
        return None

    _, answer0 = separate_reasoning_and_answer(resp)
    if not answer0.strip():
        answer0 = "<empty answer>"

    print(f"Fetched root answer {idx}")                #  <── loggning direkt

    msgs.append({"role": "assistant", "content": answer0})
    return AttemptRoot(msgs, answer0)

def _format_log(idx: int, root_ans: str, colors: list[str], verdict: str) -> str:
    return (
        f"\nAttempt {idx}"
        f"\n-------"
        f"\nInitial answer: {root_ans}"
        f"\nStated colors: {', '.join(colors) if colors else '—'}"
        f"\n==> {verdict}"
    )

async def evaluate_attempt(
    model: str, root: AttemptRoot, idx: int, sem: asyncio.Semaphore
) -> tuple[str, list[str] | None]:
    colors: list[str] = []
    claimed: Optional[str] = None
    dialogue: list[str] = []

    for b in range(NUM_BRANCHES):
        q = "is it white" if b % 2 == 0 else "is it black"
        branch_msgs = root.msgs + [{"role": "user", "content": q}]
        resp = await chat(model, branch_msgs, sem)
        if not resp:
            return _format_log(idx, root.answer0, colors, "Error: no response"), None

        _, ans_i = separate_reasoning_and_answer(resp)
        yes_no = extract_yes_no(ans_i)
        color = determine_claimed_color(q, yes_no)
        if color is None:
            return _format_log(idx, root.answer0, colors, "Error: could not parse yes/no"), None

        colors.append(color)
        dialogue.append(f"Reply {b+1}: {color}")

        if claimed is None:
            claimed = color
        elif claimed != color:
            return _format_log(idx, root.answer0, colors, "Inconsistent answers"), None

    return _format_log(idx, root.answer0, colors, f"Consistent answers ({claimed})"), dialogue

# ───────────────────────── model-testare (async) ─────────────────────
async def test_model(model: str, attempts: int) -> list[list[str]]:
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    print(f"\n========== {model.split('/')[-1]} ==========")

    roots = await asyncio.gather(
        *[asyncio.create_task(get_root_reply(model, i + 1, sem)) for i in range(attempts)]
    )
    eval_tasks = [
        asyncio.create_task(evaluate_attempt(model, root, idx + 1, sem))
        for idx, root in enumerate(r for r in roots if r)
    ]
    results = await asyncio.gather(*eval_tasks)

    winners: list[list[str]] = []
    for log_block, dialogue in results:
        print(log_block)
        if dialogue:
            winners.append(dialogue)
    return winners

# ─────────────────────────── huvudprogram ────────────────────────────
async def main_async() -> None:
    all_winners: list[list[str]] = []
    for m in MODELS:
        all_winners.extend(await test_model(m, NUM_ATTEMPTS))

    with open("winners.json", "w", encoding="utf-8") as f:
        json.dump(all_winners, f, indent=2)

    print(f"\nTotal successful attempts: {len(all_winners)}/{NUM_ATTEMPTS}")

if __name__ == "__main__":
    asyncio.run(main_async())
