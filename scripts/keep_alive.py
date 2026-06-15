"""
keep_alive.py — open the deployed Streamlit app in a real browser session.

Plain HTTP pings do not wake Streamlit Community Cloud apps. This script uses
Playwright so JavaScript runs and the WebSocket connects — the same as a human
visitor clicking through the sleep screen.

Streamlit Cloud wraps the app in an iframe (title="streamlitApp"); selectors
must target that frame, not the top-level page.
"""

import os
import re
import sys

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

APP_URL = os.environ.get(
    "STREAMLIT_APP_URL",
    "https://self-verifying-assistant.streamlit.app",
)
TIMEOUT_MS = int(os.environ.get("KEEP_ALIVE_TIMEOUT_MS", "180000"))


def main() -> int:
    print(f"Opening {APP_URL}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(APP_URL, wait_until="load", timeout=TIMEOUT_MS)
            page.wait_for_selector('iframe[title="streamlitApp"]', timeout=TIMEOUT_MS)
            app = page.frame_locator('iframe[title="streamlitApp"]')

            try:
                wake = app.get_by_role(
                    "button", name=re.compile(r"get this app back up", re.I)
                )
                wake.first.wait_for(state="visible", timeout=15_000)
                print("App asleep — clicking wake button")
                wake.first.click(timeout=30_000)
            except PlaywrightTimeout:
                pass

            app.locator('[data-testid="stApp"]').wait_for(timeout=TIMEOUT_MS)
            print("Streamlit shell loaded")

            # Let the WebSocket session stay open briefly so traffic counts.
            page.wait_for_timeout(10_000)
        except PlaywrightTimeout as exc:
            print(f"Keep-alive failed: {exc}", file=sys.stderr)
            return 1
        finally:
            browser.close()

    print("Keep-alive OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
