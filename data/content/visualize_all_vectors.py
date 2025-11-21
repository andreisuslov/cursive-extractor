import json
import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

# Configuration
PDF_PATH = "data/content/test_document.pdf"
JSON_PATH = "data/content/bbox_data/page_2_data_with_vectors.json"
OUTPUT_IMAGE_PATH = "data/content/page_2_vectorized_overlay.jpg"
TARGET_PAGE_INDEX = 1  # Page 2 (0-indexed)

def visualize_all_vectors():
    # 1. Load the PDF Page
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: {PDF_PATH} not found.")
        return

    print(f"📖 Reading PDF...")
    try:
        # Use 600 DPI for high quality
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

    # Convert to OpenCV format
    open_cv_image = np.array(page_image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()

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

    print(f"Processing {len(data)} entries...")

    # 3. Draw overlays for each word
    overlay_count = 0
    
    for i, entry in enumerate(data):
        if "box_2d" not in entry or "points" not in entry:
            continue
            
        box_2d = entry["box_2d"]
        points = entry["points"]
        
        if not points:
            continue
        
        # [y_min, x_min, y_max, x_max] (1000-scale)
        scale_x = width / 1000.0
        scale_y = height / 1000.0
        
        y_min = int(box_2d[0] * scale_y)
        x_min = int(box_2d[1] * scale_x)
        y_max = int(box_2d[2] * scale_y)
        x_max = int(box_2d[3] * scale_x)
        
        # Add padding
        padding = 10
        box_x_min = max(0, x_min - padding)
        box_y_min = max(0, y_min - padding)
        box_width = min(width, x_max + padding) - box_x_min
        box_height = min(height, y_max + padding) - box_y_min
        
        # Draw the vectorized strokes
        for j in range(len(points) - 1):
            p1 = points[j]
            p2 = points[j + 1]
            
            # Convert normalized coordinates to pixel coordinates
            # Normalized coords are relative to the cropped box
            x1 = int(box_x_min + p1[0] * box_width)
            y1 = int(box_y_min + p1[1] * box_height)
            x2 = int(box_x_min + p2[0] * box_width)
            y2 = int(box_y_min + p2[1] * box_height)
            
            # Only draw if pen is down (state == 1)
            if p1[2] == 1 and p2[2] == 1:
                cv2.line(open_cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        overlay_count += 1
        if overlay_count % 10 == 0:
            print(f"Drew overlays for {overlay_count}/{len(data)} words...")

    # 4. Save the result
    print(f"✅ Drew overlays for {overlay_count} words.")
    print(f"💾 Saving to {OUTPUT_IMAGE_PATH}...")
    cv2.imwrite(OUTPUT_IMAGE_PATH, open_cv_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print("✅ Done.")
    print(f"📍 Output saved to: {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    visualize_all_vectors()
