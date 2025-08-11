from __future__ import annotations

import asyncio
import os

# Load environment variables from a .env file
from dotenv import load_dotenv
load_dotenv()

from .undetected_browser import BrowserConfig, launch_stealth_browser, open_new_page, graceful_close


MOTIONS_URL = "https://app.hera.video/motions"


async def simple_google_login() -> None:
    config = BrowserConfig(headless=False, slow_mo_ms=300)
    pw, browser, context = await launch_stealth_browser(config)
    try:
        page = await open_new_page(context)
        await page.goto(MOTIONS_URL, wait_until="domcontentloaded")
        
        gmail = os.getenv("GMAIL")
        gpass = os.getenv("GPASS")
        
        if not gmail or not gpass:
            print("Error: GMAIL and GPASS must be set in environment or .env file")
            return

        # Step 1: Click Google sign-in button
        google_btn = page.locator("xpath=/html/body/div[3]/div[2]/button")
        await google_btn.click()

        # Step 2: Fill email
        email_input = page.locator("xpath=/html/body/div[2]/div[1]/div[2]/c-wiz/main/div[2]/div/div/div[1]/form/span/section/div/div/div[1]/div[1]/div[1]/div/div[1]/input")
        await email_input.fill(gmail)

        # Step 3: Click Next (after email)
        next_btn_1 = page.locator("xpath=/html/body/div[2]/div[1]/div[2]/c-wiz/main/div[3]/div/div[1]/div/div/button")
        await next_btn_1.click()

        # Step 4: Fill password
        password_input = page.locator("xpath=/html/body/div[2]/div[1]/div[2]/c-wiz/main/div[2]/div/div/div[1]/form/span/section[2]/div/div/div[1]/div[1]/div/div/div/div/div[1]/div/div[1]/input")
        await password_input.fill(gpass)

        # Step 5: Click Next (after password)
        next_btn_2 = page.locator("xpath=/html/body/div[2]/div[1]/div[2]/c-wiz/main/div[3]/div/div[1]/div/div/button")
        await next_btn_2.click()

        # Wait until user closes the window
        await page.wait_for_event("close", timeout=0)
    finally:
        await graceful_close(pw, browser)


def main() -> None:
    # Windows-specific: ensure subprocess support (required by Playwright)
    try:
        import asyncio
        if hasattr(asyncio, "WindowsProactorEventLoopPolicy"):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

    asyncio.run(simple_google_login())


if __name__ == "__main__":
    main()


