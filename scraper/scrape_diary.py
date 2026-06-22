"""Scrape an American Diary Project diary's pages into a single PDF.

A headless Selenium Chrome session loads each gallery page, downloads the
full-resolution scan, and assembles the images into one PDF under ``outputs/``
(named from the diary slug + page range). Works for ANY diary on
americandiaryproject.com: pass its collection page-gallery URL with ``--url``.
The total page count is auto-detected from the gallery unless you pass
``--pages`` (a count) and/or ``--start`` (1-based first page).
"""

import argparse
import os
import re
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

# Default diary: the 1-10-2000 black spiralbound diary from a New Yorker (82 pages).
DEFAULT_URL = "https://americandiaryproject.com/collection/1-10-2000-black-spiralbound-diary-from-a-new-yorker/"
OUTPUT_DIR = "outputs"


def diary_slug(url):
    """The collection slug from a diary URL ('.../collection/<slug>/' -> '<slug>')."""
    m = re.search(r"/collection/([a-z0-9\-]+)", url)
    return m.group(1) if m else "diary"


def detect_total_pages(url):
    """Read the gallery's reported page count (BWG embeds it as ``total-pages_0``).

    Returns the int, or None if the page can't be fetched/parsed."""
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        m = re.search(r'total-pages_0">\s*(\d+)', r.text)
        return int(m.group(1)) if m else None
    except Exception as e:
        print(f"⚠️ Could not detect page count: {e}")
        return None


def setup_driver():
    """Build a headless Chrome WebDriver (working around webdriver-manager
    occasionally returning the notices file instead of the chromedriver binary)."""
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
    os.chmod(path, 0o755)  # webdriver-manager sometimes drops the +x bit
    driver = webdriver.Chrome(service=Service(path), options=options)
    return driver


def download_image(url, filename):
    """Stream ``url`` to ``filename``; return True on success, False on any error."""
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
    """Combine every .jpg in ``image_folder`` (sorted) into one ``output_pdf``."""
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


def build_output_path(slug, start, end, total):
    """Path for the assembled PDF, INSIDE its own per-document folder.

    The document stem (diary slug + the page range it covers) names both the
    folder and the PDF, so everything for this document -- PDF, transcript, and
    the OCR pipeline's page_NNN/ folders -- lives together under one directory:
    ``outputs/<stem>/<stem>.pdf``.
    """
    if start == 1 and total and end >= total:
        stem = f"{slug}_full_1-{total}"
    else:
        stem = f"{slug}_pages_{start}-{end}"
    return os.path.join(OUTPUT_DIR, stem, f"{stem}.pdf")


def main(url=DEFAULT_URL, start=1, pages=None):
    slug = diary_slug(url)
    total = detect_total_pages(url)
    if total:
        print(f"Diary '{slug}': {total} pages reported by gallery.")
    # How many pages to grab: explicit --pages, else the rest of the diary.
    if pages is None:
        if not total:
            raise SystemExit("Could not detect page count; pass --pages N explicitly.")
        count = total - start + 1
    else:
        count = pages
    end = start + count - 1
    if total:
        end = min(end, total)
        count = end - start + 1
    if count <= 0:
        raise SystemExit(f"Empty range: start={start}, pages={pages}, total={total}")

    output_pdf = build_output_path(slug, start, end, total)
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)  # the per-document folder
    temp_folder = f"diary_images_{slug}"
    os.makedirs(temp_folder, exist_ok=True)

    driver = setup_driver()

    try:
        print(f"Downloading pages {start}..{end} of '{slug}' -> {output_pdf}")

        for page_num in range(start, end + 1):
            target_url = f"{url}?page_number_0={page_num}"

            driver.get(target_url)

            try:
                wait = WebDriverWait(driver, 10)
                img_element = wait.until(
                    EC.presence_of_element_located((By.CLASS_NAME, "bwg_image_browser_img"))
                )

                img_url = img_element.get_attribute("src")

                if img_url:
                    file_name = f"{temp_folder}/page_{page_num:03d}.jpg"
                    download_image(img_url, file_name)
                else:
                    print(f"⚠️ No image source found for page {page_num}")

            except Exception as e:
                print(f"⚠️ Could not find image on page {page_num}: {e}")

            # Be polite to the server
            time.sleep(1)

        # Create PDF after all images are downloaded
        create_pdf(temp_folder, output_pdf)

    finally:
        driver.quit()
        # Cleanup temp folder
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
            print(f"Cleaned up temporary folder: {temp_folder}")
        print("Process complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape an American Diary Project diary to PDF")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Diary collection page-gallery URL (default: the 1-10-2000 NY diary)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="First page to scrape, 1-based (default: 1)",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=None,
        help="Number of pages to scrape from --start (default: rest of the diary)",
    )
    args = parser.parse_args()
    main(url=args.url, start=args.start, pages=args.pages)
