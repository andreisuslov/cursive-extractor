import argparse
import os
import shutil
import time

import requests
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# --- CONFIGURATION ---
BASE_URL = "https://americandiaryproject.com/collection/1-10-2000-black-spiralbound-diary-from-a-new-yorker/"
TOTAL_PAGES = 82
TEMP_FOLDER = "diary_images"
OUTPUT_DIR = "outputs"


def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
    path = ChromeDriverManager().install()
    if path.endswith("THIRD_PARTY_NOTICES.chromedriver"):
        path = os.path.join(os.path.dirname(path), "chromedriver")
    driver = webdriver.Chrome(service=Service(path), options=options)
    return driver


def download_image(url, filename):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, stream=True)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"✅ Saved: {filename}")
            return True
        else:
            print(f"❌ Failed to download content: {url}")
            return False
    except Exception as e:
        print(f"❌ Error downloading: {e}")
        return False


def create_pdf(image_folder, output_pdf):
    print("Creating PDF...")
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg")])

    if not image_files:
        print("No images found to create PDF.")
        return

    images = []
    for file in image_files:
        img_path = os.path.join(image_folder, file)
        try:
            img = Image.open(img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"Error processing image {file}: {e}")

    if images:
        images[0].save(output_pdf, "PDF", resolution=100.0, save_all=True, append_images=images[1:])
        print(f"✅ PDF created successfully: {output_pdf}")
    else:
        print("No valid images to save.")


def build_output_path(total_pages):
    """Name the PDF so it signifies full vs partial content of the diary."""
    if total_pages >= TOTAL_PAGES:
        filename = f"diary_full_1-{TOTAL_PAGES}.pdf"
    else:
        filename = f"diary_partial_pages_1-{total_pages}.pdf"
    return os.path.join(OUTPUT_DIR, filename)


def main(total_pages=TOTAL_PAGES):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_pdf = build_output_path(total_pages)
    if not os.path.exists(TEMP_FOLDER):
        os.makedirs(TEMP_FOLDER)

    driver = setup_driver()

    try:
        print(f"Starting download process for {total_pages} page(s)...")

        for page_num in range(1, total_pages + 1):
            target_url = f"{BASE_URL}?page_number_0={page_num}"

            driver.get(target_url)

            try:
                wait = WebDriverWait(driver, 10)
                img_element = wait.until(
                    EC.presence_of_element_located((By.CLASS_NAME, "bwg_image_browser_img"))
                )

                img_url = img_element.get_attribute("src")

                if img_url:
                    file_name = f"{TEMP_FOLDER}/page_{page_num:02d}.jpg"
                    download_image(img_url, file_name)
                else:
                    print(f"⚠️ No image source found for page {page_num}")

            except Exception as e:
                print(f"⚠️ Could not find image on page {page_num}: {e}")

            # Be polite to the server
            time.sleep(1)

        # Create PDF after all images are downloaded
        create_pdf(TEMP_FOLDER, output_pdf)

    finally:
        driver.quit()
        # Cleanup temp folder
        if os.path.exists(TEMP_FOLDER):
            shutil.rmtree(TEMP_FOLDER)
            print(f"Cleaned up temporary folder: {TEMP_FOLDER}")
        print("Process complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape diary pages into a PDF")
    parser.add_argument(
        "--pages",
        type=int,
        default=TOTAL_PAGES,
        help=f"Number of pages to scrape, starting from page 1 (default: {TOTAL_PAGES})",
    )
    args = parser.parse_args()
    main(total_pages=args.pages)
