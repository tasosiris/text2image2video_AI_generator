from __future__ import annotations

import asyncio
import os
import json

# Load environment variables from a .env file
from dotenv import load_dotenv
load_dotenv()

from .undetected_browser import BrowserConfig, launch_stealth_browser, open_new_page, graceful_close


FASTMOSS_URL = "https://www.fastmoss.com/e-commerce/saleslist?region=US"


async def fastmoss_google_login() -> None:
    
    max_retries = 5
    for attempt in range(max_retries):
        print(f"--- Attempt {attempt + 1} of {max_retries} ---")
        config = BrowserConfig(headless=False, slow_mo_ms=300)
        pw, browser, context = await launch_stealth_browser(config)
        
        try:
            page = await open_new_page(context)
            await page.goto(FASTMOSS_URL, wait_until="domcontentloaded")

            # Step 1: Click the div to show the login page
            await page.locator("xpath=/html/body/div[1]/div/div[1]/div[2]/div[1]").click()
            await page.wait_for_timeout(2000)

            # Use Playwright's event listener to wait for the new page
            async with context.expect_page() as page_info:
                # Step 2: Click the button that opens the Google login popup
                await page.locator("xpath=/html/body/div[2]/div/div[2]/div/div[1]/div/div/div/div/div[2]/button[1]").click()

            new_page = await page_info.value
            await new_page.wait_for_load_state()
            print("Successfully detected new page for Google login.")

            # --- Google Login Flow on the new page ---
            gmail = os.getenv("GMAIL")
            gpass = os.getenv("GPASS")

            if not gmail or not gpass:
                print("Error: GMAIL and GPASS must be set in environment or .env file")
                return

            # Step 3: Fill email
            await new_page.locator("xpath=/html/body/div[2]/div[1]/div[2]/c-wiz/main/div[2]/div/div/div[1]/form/span/section/div/div/div[1]/div[1]/div[1]/div/div[1]/input").fill(gmail)
            await new_page.wait_for_timeout(1000)

            # Step 4: Click Next (after email)
            await new_page.locator("xpath=/html/body/div[2]/div[1]/div[2]/c-wiz/main/div[3]/div/div[1]/div/div/button").click()
            await new_page.wait_for_timeout(2000)

            # Step 5: Fill password
            await new_page.locator("xpath=/html/body/div[2]/div[1]/div[2]/c-wiz/main/div[2]/div/div/div[1]/form/span/section[2]/div/div/div[1]/div[1]/div/div/div/div/div[1]/div/div[1]/input").fill(gpass)
            await new_page.wait_for_timeout(1000)

            # Step 6: Click Next (after password)
            await new_page.locator("xpath=/html/body/div[2]/div[1]/div[2]/c-wiz/main/div[3]/div/div[1]/div/div/button").click()
            
            print("Login successful. Returning to original page to scrape data.")

            # Bring the main page to the front
            await page.bring_to_front()
            
            print("Waiting for data to load in the table after login...")
            # Wait for the first actual data row to appear (tr[2] based on user XPath)
            await page.locator("xpath=//table/tbody/tr[2]").wait_for(timeout=15000)
            await page.wait_for_timeout(2000) # Extra small wait for full render

            # Define output paths and create directories
            PRODUCTS_DIR = os.path.join("outputs", "products")
            PHOTOS_DIR = os.path.join(PRODUCTS_DIR, "photos")
            os.makedirs(PHOTOS_DIR, exist_ok=True)
            
            print("Scraping data from the table...")
            try:
                table_body = page.locator("xpath=//table/tbody")
                await table_body.wait_for(timeout=10000) # Wait for table to be present

                # Get headers
                table = table_body.locator("xpath=..")
                header_elements = await table.locator("thead th").all()
                headers = [await h.text_content() for h in header_elements]
                headers = [h.strip() for h in headers if h]

                scraped_data = []
                rows = await table_body.locator("tr").all()
                
                print(f"Found {len(rows)} rows. Scraping...")

                for i, row in enumerate(rows):
                    row_data = {}
                    cells = await row.locator("td").all()
                    
                    # Process all cells for their text content first
                    for j, cell in enumerate(cells):
                        header = headers[j] if j < len(headers) else f"column_{j+1}"
                        cell_text = await cell.text_content()
                        row_data[header] = cell_text.strip() if cell_text else ""
                    
                    # Handle image download from the second cell (index 1)
                    if len(cells) > 1:
                        # Find the image, which might be nested deep within the cell
                        img_element = cells[1].locator("xpath=.//img")
                        if await img_element.count() > 0:
                            img_src = await img_element.get_attribute("src")
                            if img_src and img_src.startswith('http'):
                                try:
                                    response = await page.request.get(img_src)
                                    if response.ok:
                                        image_data = await response.body()
                                        file_extension = os.path.splitext(img_src.split('?')[0])[-1] or '.jpg'
                                        if len(file_extension) > 5 or len(file_extension) < 2: file_extension = '.jpg'
                                        filename = f"product_{i}{file_extension}"
                                        filepath = os.path.join(PHOTOS_DIR, filename)
                                        with open(filepath, "wb") as img_file:
                                            img_file.write(image_data)
                                        row_data["product_image_filename"] = filename
                                    else:
                                        row_data["product_image_filename"] = f"download_failed_status_{response.status}"
                                except Exception as e:
                                    print(f"Could not download image {img_src}. Error: {e}")
                                    row_data["product_image_filename"] = "download_error"
                            else:
                                row_data["product_image_filename"] = "image_src_not_found"
                        else:
                            row_data["product_image_filename"] = "image_tag_not_found"
                            
                    scraped_data.append(row_data)

                # --- Post-processing ---
                # 1. Remove the first empty object
                if scraped_data:
                    scraped_data.pop(0)

                # 2. Remove the 'Action' key from all objects and add image filename
                for item in scraped_data:
                    item.pop("Action", None)
                
                # Save to JSON file in the specified output directory
                json_output_path = os.path.join(PRODUCTS_DIR, "products_data.json")
                with open(json_output_path, "w", encoding="utf-8") as f:
                    json.dump(scraped_data, f, indent=4, ensure_ascii=False)

                print(f"Successfully saved table data to {json_output_path}")

            except Exception as e:
                print(f"Could not scrape the table. Error: {e}")
            
            print("Scraping complete. Waiting for 10 minutes.")
            await page.wait_for_timeout(600000) # Keep browser open for 10 minutes
            
            # If we get here, it means success, so break the loop
            break

        except Exception as e:
            print(f"An error occurred or the second page was not found: {e}")
            if attempt < max_retries - 1:
                print("Closing browser and restarting...")
            else:
                print("Max retries reached. Exiting.")
        
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
    
    try:
        asyncio.run(fastmoss_google_login())
    except KeyboardInterrupt:
        print("\nScript stopped by user.")


if __name__ == "__main__":
    main()
