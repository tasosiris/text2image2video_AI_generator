from __future__ import annotations

import asyncio
import os
import json
from PIL import Image
import io
import re
import hashlib

# Load environment variables from a .env file
from dotenv import load_dotenv
load_dotenv()

from .undetected_browser import BrowserConfig, launch_stealth_browser, open_new_page, graceful_close


FASTMOSS_URL = "https://www.fastmoss.com/e-commerce/saleslist?region=US"


def generate_stable_product_id(product_name_raw: str, category: str) -> str:
    """Generates a stable, unique ID from a product name using a hash."""
    # Extract a cleaner name to create a stable base for the hash
    if "Price:" in product_name_raw:
        name = product_name_raw.split("Price:")[0].strip()
    else:
        # Fallback for long names without a clear delimiter
        name = product_name_raw[:80].strip()
    
    # Normalize the string: lowercase, remove extra whitespace
    normalized_name = name.lower().strip()
    normalized_name = re.sub(r'\s+', ' ', normalized_name)
    
    # Create a SHA-1 hash of the normalized name
    hasher = hashlib.sha1(normalized_name.encode('utf-8'))
    
    # Take the first 12 characters of the hex digest for a short but unique ID
    short_hash = hasher.hexdigest()[:12]
    
    return f"{category.lower()}_{short_hash}"


async def fastmoss_google_login() -> None:
    
    max_retries = 5
    for attempt in range(max_retries):
        print(f"--- Attempt {attempt + 1} of {max_retries} ---")
        config = BrowserConfig(headless=False, slow_mo_ms=100)
        pw, browser, context = await launch_stealth_browser(config)
        
        try:
            page = await open_new_page(context)
            await page.goto(FASTMOSS_URL, wait_until="domcontentloaded")

            # --- PRE-SCRAPE SETUP ---
            PRODUCTS_DIR = os.path.join("outputs", "products")
            final_output_path = os.path.join(PRODUCTS_DIR, "products_data.json")
            
            # Load existing products to avoid duplicates
            existing_products = {}
            if os.path.exists(final_output_path):
                print(f"Found existing product data file at '{final_output_path}'. Loading to prevent duplicates.")
                with open(final_output_path, "r", encoding="utf-8") as f:
                    try:
                        all_products = json.load(f)
                        existing_products = {p['product_id']: p for p in all_products}
                        print(f"Loaded {len(existing_products)} existing products.")
                    except json.JSONDecodeError:
                        print("Warning: Could not parse existing product data file. Starting fresh.")
                        all_products = []
            else:
                all_products = []

            # Get a set of existing IDs for quick lookups
            existing_ids = set(existing_products.keys())

            # Step 1: Click the div to show the login page
            await page.locator("xpath=/html/body/div[1]/div/div[1]/div[2]/div[1]").click()
            await page.wait_for_timeout(500)

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
            await new_page.wait_for_timeout(300)

            # Step 4: Click Next (after email)
            await new_page.locator("xpath=/html/body/div[2]/div[1]/div[2]/c-wiz/main/div[3]/div/div[1]/div/div/button").click()
            await new_page.wait_for_timeout(500)

            # Step 5: Fill password
            await new_page.locator("xpath=/html/body/div[2]/div[1]/div[2]/c-wiz/main/div[2]/div/div/div[1]/form/span/section[2]/div/div/div[1]/div[1]/div/div/div/div/div[1]/div/div[1]/input").fill(gpass)
            await new_page.wait_for_timeout(300)

            # Step 6: Click Next (after password)
            await new_page.locator("xpath=/html/body/div[2]/div[1]/div[2]/c-wiz/main/div[3]/div/div[1]/div/div/button").click()
            
            print("Login successful. Returning to original page to scrape data.")

            # Bring the main page to the front
            await page.bring_to_front()
            
            # --- Scrape Beauty Products ---
            BEAUTY_DIR = os.path.join(PRODUCTS_DIR, "Beauty")
            
            print("--- Navigating to Beauty category page ---")
            await page.goto("https://www.fastmoss.com/e-commerce/saleslist?region=US&page=1&l1_cid=14", wait_until="domcontentloaded")
            await page.wait_for_timeout(3000) # Wait for content to load

            new_beauty_products = await scrape_products(page, "Beauty", BEAUTY_DIR, existing_ids)
            if new_beauty_products:
                all_products.extend(new_beauty_products)
                # Update existing_ids with the new ones to avoid intra-run duplicates
                for p in new_beauty_products:
                    existing_ids.add(p['product_id'])

            # --- Scrape Health Products ---
            HEALTH_DIR = os.path.join(PRODUCTS_DIR, "Health")

            print("--- Navigating to Health category page ---")
            await page.goto("https://www.fastmoss.com/e-commerce/search?region=US&page=1&l1_cid=25", wait_until="domcontentloaded")
            await page.wait_for_timeout(3000) # Wait for content to load
            
            new_health_products = await scrape_products(page, "Health", HEALTH_DIR, existing_ids)
            if new_health_products:
                all_products.extend(new_health_products)

            # --- Save all products to a single file ---
            if all_products:
                with open(final_output_path, "w", encoding="utf-8") as f:
                    json.dump(all_products, f, indent=4, ensure_ascii=False)
                print(f"\nSuccessfully saved a total of {len(all_products)} products to '{final_output_path}'")
            else:
                print("\nNo products were scraped from any category.")

            print("Scraping for all categories complete.")
            await page.wait_for_timeout(5000) # Keep browser open for a short while
            
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


async def scrape_products(page, category_name: str, output_dir: str, existing_ids: set) -> list | None:
    """
    Scrapes product data and images from the currently loaded table.
    Saves images for new products and returns their structured data.
    """
    print(f"--- Scraping '{category_name}' products into '{output_dir}' ---")

    # Define output paths and create directories
    PHOTOS_DIR = os.path.join(output_dir, "photos")
    os.makedirs(PHOTOS_DIR, exist_ok=True)
    
    newly_scraped_data = []
    print("Scraping data from the table...")
    try:
        # Wait for the table to be present and ready
        table_body = page.locator("xpath=//table/tbody")
        await table_body.wait_for(timeout=10000)
        
        # Wait for at least one data row to ensure content is loaded
        await page.locator("xpath=//table/tbody/tr[2]").wait_for(timeout=15000)
        await page.wait_for_timeout(2000) # Extra small wait for full render

        # Get headers
        table = table_body.locator("xpath=..")
        header_elements = await table.locator("thead th").all()
        headers = [await h.text_content() for h in header_elements]
        headers = [h.strip() for h in headers if h]

        rows = await table_body.locator("tr").all()
        
        print(f"Found {len(rows)} rows. Checking for new products...")

        for row in rows:
            row_data = {}
            cells = await row.locator("td").all()
            
            # Process all cells for their text content first
            for j, cell in enumerate(cells):
                header = headers[j] if j < len(headers) else f"column_{j+1}"
                cell_text = await cell.text_content()
                row_data[header] = cell_text.strip() if cell_text else ""
            
            # Generate stable ID and check for duplicates BEFORE downloading image
            product_name_raw = row_data.get("Products", "")
            if not product_name_raw:
                continue
            
            product_id = generate_stable_product_id(product_name_raw, category_name)
            
            if product_id in existing_ids:
                continue # Skip this product, it's a duplicate

            # Handle image download for the new product
            try:
                img_element = row.locator("xpath=.//img").first
                if await img_element.count() > 0:
                    img_src = await img_element.get_attribute("src")
                    if img_src and img_src.startswith('http'):
                        try:
                            response = await page.request.get(img_src)
                            if response.ok:
                                image_data = await response.body()

                                # --- Image Resizing Logic ---
                                try:
                                    with Image.open(io.BytesIO(image_data)) as img:
                                        original_width, original_height = img.size
                                        target_aspect = 9 / 16
                                        new_width = original_width
                                        new_height = int(new_width / target_aspect)
                                        new_img = Image.new("RGB", (new_width, new_height), "black")
                                        paste_y = (new_height - original_height) // 2
                                        new_img.paste(img, (0, paste_y))
                                        buffer = io.BytesIO()
                                        new_img.save(buffer, format='JPEG')
                                        image_data = buffer.getvalue()
                                except Exception as e:
                                    print(f"Failed to resize image {img_src}. Error: {e}")
                                # --- End Image Resizing ---

                                file_extension = os.path.splitext(img_src.split('?')[0])[-1] or '.jpg'
                                if len(file_extension) > 5 or len(file_extension) < 2: file_extension = '.jpg'
                                
                                filename = f"{product_id}{file_extension}"
                                filepath = os.path.join(PHOTOS_DIR, filename)
                                
                                with open(filepath, "wb") as img_file:
                                    img_file.write(image_data)
                                
                                # Add new/updated fields
                                row_data["product_id"] = product_id
                                row_data["category"] = category_name
                                row_data["videos_generated"] = 0
                                row_data["product_image_filename"] = f"{category_name}/photos/{filename}"
                                
                                newly_scraped_data.append(row_data) # Add to list of new products

                            else:
                                print(f"Image download failed with status {response.status} for {product_id}")
                        except Exception as e:
                            print(f"Could not download image for {product_id}. Error: {e}")
            except Exception as e:
                print(f"Error processing images for a new product: {e}")
        
        # --- Post-processing ---
        if newly_scraped_data:
            # Clean up the "Action" column if it exists
            for item in newly_scraped_data:
                item.pop("Action", None)
            print(f"Found and processed {len(newly_scraped_data)} new products in '{category_name}'.")
        
        return newly_scraped_data

    except Exception as e:
        print(f"Could not scrape the table in {output_dir}. Error: {e}")
        return None


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
