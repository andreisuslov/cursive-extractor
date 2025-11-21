import json
import os
from pdf2image import convert_from_path
from PIL import Image, ImageDraw

# Configuration
PDF_PATH = "data/content/test_document.pdf"
JSON_PATH = "data/content/handwriting_dataset_precise.json"
OUTPUT_IMAGE_PATH = "data/content/first_entry_visualized.jpg"
TARGET_PAGE_INDEX = 1  # Page 2

def visualize_first_entry():
    # 1. Load the PDF Page
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: {PDF_PATH} not found.")
        return

    print(f"📖 Reading PDF...")
    try:
        images = convert_from_path(PDF_PATH)
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
    print(f"✅ Loaded First Entry.")
    print(f"Metadata: {json.dumps(first_entry.get('metadata', {}), indent=2)}")

    # 3. Analyze Points
    points = first_entry.get("points", [])
    if not points:
        print("❌ No points in first entry.")
        return

    print(f"Number of points: {len(points)}")
    
    # Calculate pixel coordinates
    pixel_points = []
    min_x, min_y = width, height
    max_x, max_y = 0, 0

    for pt in points:
        x_norm, y_norm, p = pt
        x = x_norm * width
        y = y_norm * height
        pixel_points.append((x, y, p))
        
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    print(f"Bounding Box of Strokes (Pixels): [{min_x:.2f}, {min_y:.2f}, {max_x:.2f}, {max_y:.2f}]")
    
    # 4. Draw Strokes
    draw = ImageDraw.Draw(page_image)
    STROKE_COLOR = (0, 255, 0) 
    STROKE_WIDTH = 3

    for i in range(1, len(pixel_points)):
        end_x, end_y, end_p = pixel_points[i]
        if end_p == 1:
            draw.line([(pixel_points[i-1][0], pixel_points[i-1][1]), (end_x, end_y)], fill=STROKE_COLOR, width=STROKE_WIDTH)

    # 5. Crop to the area (with padding)
    padding = 50
    crop_box = (
        max(0, min_x - padding),
        max(0, min_y - padding),
        min(width, max_x + padding),
        min(height, max_y + padding)
    )
    
    cropped_image = page_image.crop(crop_box)
    
    # 6. Save Image
    cropped_image.save(OUTPUT_IMAGE_PATH)
    print(f"🖼️ Visualized First Entry saved to: {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    visualize_first_entry()
