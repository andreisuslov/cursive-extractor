import json
import os
from pdf2image import convert_from_path
from PIL import Image

# Configuration
PDF_PATH = "data/content/test_document.pdf"
JSON_PATH = "data/content/bbox_data/page_2_data.json"
OUTPUT_IMAGE_PATH = "data/content/temp_crop.jpg"
TARGET_PAGE_INDEX = 1  # Page 2 (0-indexed)

def generate_first_box_image():
    # 1. Load the PDF Page at high DPI for quality
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: {PDF_PATH} not found.")
        return

    print(f"📖 Reading PDF...")
    try:
        # Use 600 DPI for ultra high quality
        images = convert_from_path(PDF_PATH, dpi=600)
        if len(images) <= TARGET_PAGE_INDEX:
            print(f"❌ PDF has fewer pages than index {TARGET_PAGE_INDEX}.")
            return
        
        page_image = images[TARGET_PAGE_INDEX]
        width, height = page_image.size
        print(f"✅ Loaded Page {TARGET_PAGE_INDEX + 1} ({width}x{height})")
    except Exception as e:
        print(f"❌ Error converting PDF: {e}")
        return

    # 2. Load the JSON Data
    if not os.path.exists(JSON_PATH):
        print(f"❌ Error: {JSON_PATH} not found.")
        return

    print(f"📖 Reading JSON Data...")
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    
    if not data:
        print("❌ Dataset is empty.")
        return

    first_entry = data[0]
    print(f"✅ Loaded First Entry: {first_entry}")

    # 3. Extract Bounding Box
    if "box_2d" not in first_entry:
        print("❌ 'box_2d' key not found in entry.")
        return

    box_2d = first_entry["box_2d"]
    # User confirmed "Blue Box" interpretation: [y_min, x_min, y_max, x_max] (1000-scale)
    
    scale_x = width / 1000.0
    scale_y = height / 1000.0
    
    y_min = int(box_2d[0] * scale_y)
    x_min = int(box_2d[1] * scale_x)
    y_max = int(box_2d[2] * scale_y)
    x_max = int(box_2d[3] * scale_x)
    
    print(f"Original Box (1000-scale YXYX): {box_2d}")
    print(f"Scaled Box (Pixels): [{x_min}, {y_min}, {x_max}, {y_max}]")

    # 4. Crop the Image
    # Add a small padding if desired
    padding = 10
    crop_box = (
        max(0, x_min - padding),
        max(0, y_min - padding),
        min(width, x_max + padding),
        min(height, y_max + padding)
    )
    
    cropped_image = page_image.crop(crop_box)
    
    # 5. Enhance Image
    from PIL import ImageEnhance
    
    # Increase contrast
    enhancer = ImageEnhance.Contrast(cropped_image)
    cropped_image = enhancer.enhance(2.0)  # Increase contrast by factor of 2
    
    # 6. Save Image
    cropped_image.save(OUTPUT_IMAGE_PATH, quality=95, subsampling=0)
    print(f"🖼️ High quality (600 DPI) & High Contrast crop saved to: {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    generate_first_box_image()
