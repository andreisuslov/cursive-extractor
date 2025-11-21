import json
import os
import google.generativeai as genai
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
import numpy as np

# Configuration
PDF_PATH = "data/content/test_document.pdf"
BBOX_DATA_PATH = "data/content/bbox_data/page_2_data.json"
OUTPUT_JSON_PATH = "data/content/first_box_strokes.json"
OUTPUT_IMAGE_PATH = "data/content/first_box_overlay.jpg"
TARGET_PAGE_INDEX = 1  # Page 2

def get_api_key():
    # Try getting API key from environment
    return os.environ.get("GOOGLE_API_KEY")

def generate_strokes():
    # 1. Load Data
    if not os.path.exists(PDF_PATH) or not os.path.exists(BBOX_DATA_PATH):
        print("❌ Files not found.")
        return

    print("📖 Loading Data...")
    with open(BBOX_DATA_PATH, 'r') as f:
        bbox_data = json.load(f)
    
    if not bbox_data:
        print("❌ No bbox data found.")
        return

    # Focus on the first box
    target_entry = bbox_data[0]
    text = target_entry.get("text")
    box = target_entry.get("box_2d") # [ymin, xmin, ymax, xmax] normalized 0-1000
    
    print(f"🎯 Processing First Box: '{text}'")
    print(f"   Box: {box}")

    # 2. Load and Crop Image
    images = convert_from_path(PDF_PATH)
    page_image = images[TARGET_PAGE_INDEX]
    width, height = page_image.size
    
    ymin, xmin, ymax, xmax = box
    # Normalize 0-1000 -> 0-1 -> pixels
    left = (xmin / 1000) * width
    top = (ymin / 1000) * height
    right = (xmax / 1000) * width
    bottom = (ymax / 1000) * height
    
    # Add small padding for context
    padding = 10
    crop_box = (
        max(0, left - padding),
        max(0, top - padding),
        min(width, right + padding),
        min(height, bottom + padding)
    )
    
    crop_img = page_image.crop(crop_box)
    crop_path = "data/content/temp_crop.jpg"
    crop_img.save(crop_path)
    print(f"✅ Created crop: {crop_path} ({crop_img.size})")

    # 3. Call Gemini
    api_key = get_api_key()
    if not api_key:
        print("❌ GOOGLE_API_KEY not found. Please set it in your environment.")
        # Create a dummy file for testing visualization if API fails
        # return 
    
    if api_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro', generation_config={"response_mime_type": "application/json"})
        
        prompt = f"""
        Task: Trace the handwriting for the text "{text}" in this image.
        
        Instructions:
        1. Identify the strokes that make up each character in "{text}".
        2. Return a list of points [x, y, p] for the strokes.
        3. x and y should be normalized coordinates (0 to 1) relative to the image provided.
        4. p is the pen state: 1 for touching paper (drawing), 0 for lifted (moving).
        5. Ensure the strokes follow the path of the handwriting precisely.
        6. Separate characters or strokes with a p=0 point if needed.
        
        JSON Schema:
        {{
            "text": "{text}",
            "strokes": [
                [x, y, p],
                [x, y, p],
                ...
            ]
        }}
        """
        
        print("🤖 Sending to Gemini...")
        try:
            response = model.generate_content([prompt, crop_img])
            result = json.loads(response.text)
            print("✅ Received response from Gemini.")
        except Exception as e:
            print(f"❌ Error calling Gemini: {e}")
            return
    else:
        print("⚠️ Skipping Gemini call (no key).")
        return

    # 4. Save Result
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"💾 Saved strokes to: {OUTPUT_JSON_PATH}")

    # 5. Visualize
    draw = ImageDraw.Draw(crop_img)
    strokes = result.get("strokes", [])
    
    crop_w, crop_h = crop_img.size
    
    # Convert normalized points to pixel points on the crop
    pixel_points = []
    for pt in strokes:
        x_norm, y_norm, p = pt
        x = x_norm * crop_w
        y = y_norm * crop_h
        pixel_points.append((x, y, p))
        
    # Draw
    STROKE_COLOR = (0, 255, 0)
    STROKE_WIDTH = 2
    
    for i in range(1, len(pixel_points)):
        end_x, end_y, end_p = pixel_points[i]
        if end_p == 1:
             draw.line([(pixel_points[i-1][0], pixel_points[i-1][1]), (end_x, end_y)], fill=STROKE_COLOR, width=STROKE_WIDTH)

    crop_img.save(OUTPUT_IMAGE_PATH)
    print(f"🖼️ Overlay saved to: {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    generate_strokes()
