import json
import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance

# Configuration
PDF_PATH = "data/content/test_document.pdf"
JSON_PATH = "data/content/bbox_data/page_2_data.json"
OUTPUT_JSON_PATH = "data/content/bbox_data/page_2_data_with_vectors.json"
TARGET_PAGE_INDEX = 1  # Page 2 (0-indexed)

def skeletonize(img):
    """
    Perform skeletonization using morphological operations.
    img: Binary image (white foreground, black background)
    """
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    
    # Use a cross-shaped structuring element
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    temp_img = img.copy()
    done = False
    
    while not done:
        eroded = cv2.erode(temp_img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(temp_img, temp)
        skel = cv2.bitwise_or(skel, temp)
        temp_img = eroded.copy()
        
        zeros = size - cv2.countNonZero(temp_img)
        if zeros == size:
            done = True
            
    return skel

def order_points(skel):
    """
    Attempt to order the skeleton pixels into a path.
    This is a naive implementation using Nearest Neighbor.
    """
    # Get all coordinates of non-zero pixels
    y_idxs, x_idxs = np.nonzero(skel)
    points = list(zip(x_idxs, y_idxs))
    
    if not points:
        return []
    
    # Start from the top-left-most point (heuristic for English)
    points.sort(key=lambda p: (p[0], p[1]))
    
    ordered_path = []
    current_point = points[0]
    ordered_path.append(current_point)
    points.remove(current_point)
    
    while points:
        # Find nearest neighbor to current_point
        # Search for points within distance 2 (adjacent or diagonal)
        candidates = [p for p in points if max(abs(p[0]-current_point[0]), abs(p[1]-current_point[1])) <= 1]
        
        if candidates:
            # Pick the first one
            next_point = candidates[0]
            ordered_path.append(next_point)
            points.remove(next_point)
            current_point = next_point
        else:
            # Jump to the next available point (discontinuity / new stroke)
            next_point = points[0]
            ordered_path.append(next_point)
            points.remove(next_point)
            current_point = next_point
            
    return ordered_path

def process_all_words():
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

    processed_count = 0
    
    for i, entry in enumerate(data):
        if "box_2d" not in entry:
            continue
            
        box_2d = entry["box_2d"]
        # [y_min, x_min, y_max, x_max] (1000-scale)
        
        scale_x = width / 1000.0
        scale_y = height / 1000.0
        
        y_min = int(box_2d[0] * scale_y)
        x_min = int(box_2d[1] * scale_x)
        y_max = int(box_2d[2] * scale_y)
        x_max = int(box_2d[3] * scale_x)
        
        # Crop with padding
        padding = 10
        crop_box = (
            max(0, x_min - padding),
            max(0, y_min - padding),
            min(width, x_max + padding),
            min(height, y_max + padding)
        )
        
        cropped_image = page_image.crop(crop_box)
        crop_width, crop_height = cropped_image.size
        
        # Enhance Contrast
        enhancer = ImageEnhance.Contrast(cropped_image)
        cropped_image = enhancer.enhance(2.0)
        
        # Convert to OpenCV format
        # PIL RGB -> OpenCV BGR -> Grayscale
        open_cv_image = np.array(cropped_image)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        
        # Preprocessing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Skeletonization
        skeleton = skeletonize(closed)
        
        # Vectorization
        path_points = order_points(skeleton)
        
        # Format points: [[x, y, state], ...]
        # State: 1 = pen down, 0 = pen up (end of stroke)
        formatted_points = []
        
        if len(path_points) > 0:
            current_stroke = []
            
            for j, p in enumerate(path_points):
                # Normalize coordinates
                nx = float(p[0]) / crop_width
                ny = float(p[1]) / crop_height
                
                # Ensure within 0-1 bounds
                nx = max(0.0, min(1.0, nx))
                ny = max(0.0, min(1.0, ny))
                
                # Round to 4 decimal places
                nx = round(nx, 4)
                ny = round(ny, 4)
                
                if j == 0:
                    current_stroke.append([nx, ny, 1])
                    continue
                
                prev_p = path_points[j-1]
                dist = max(abs(p[0]-prev_p[0]), abs(p[1]-prev_p[1]))
                
                if dist <= 2:
                    # Connected
                    current_stroke.append([nx, ny, 1])
                else:
                    # Jump - End of previous stroke
                    # Add end marker to current stroke
                    last_p = current_stroke[-1]
                    current_stroke.append([last_p[0], last_p[1], 0])
                    
                    # Add current stroke to formatted_points
                    formatted_points.extend(current_stroke)
                    
                    # Start new stroke
                    current_stroke = [[nx, ny, 1]]
            
            # Finish last stroke
            if current_stroke:
                last_p = current_stroke[-1]
                current_stroke.append([last_p[0], last_p[1], 0])
                formatted_points.extend(current_stroke)
        
        # Update Entry
        entry["points"] = formatted_points
        entry["metadata"] = {
            "author": "robot",
            "asciiSequence": entry.get("text", ""),
            "pointCount": len(formatted_points),
            "strokeCount": sum(1 for p in formatted_points if p[2] == 0),
            "aspectRatio": round(crop_width / crop_height, 4)
        }
        
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"Processed {processed_count}/{len(data)} words...")

    # 3. Save Updated JSON
    print(f"✅ Processed {processed_count} words.")
    print(f"💾 Saving to {OUTPUT_JSON_PATH}...")
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(data, f, indent=4)
    print("✅ Done.")

if __name__ == "__main__":
    process_all_words()
