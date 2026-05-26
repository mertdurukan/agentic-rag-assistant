"""Capture a Gradio UI screenshot — the demo image embedded in the README.

Usage:        python -m scripts.screenshot_ui
Prerequisite: uvicorn must be running on 127.0.0.1:8000 + ingestion must be done.
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8000/"
QUESTION = "What are training-free looped transformers?"
OUT = Path("docs/screenshot_gradio.png")


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1280, "height": 860})
        page = ctx.new_page()
        page.goto(URL, wait_until="networkidle")

        # Gradio Blocks: textbox + radio + Ask button
        textbox = page.locator("textarea").first
        textbox.fill(QUESTION)

        # "Ask" button click
        page.get_by_role("button", name="Ask").click()

        # Wait for the answer to load (until the faithful badge or a source link
        # appears). Timeout is relatively long because the end-to-end pipeline
        # takes ~5-8 s.
        page.wait_for_selector("text=faithful", timeout=60_000)
        # Short wait for the render to fully settle
        page.wait_for_timeout(800)

        page.screenshot(path=str(OUT), full_page=True)
        browser.close()

    print(f"Saved: {OUT.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
