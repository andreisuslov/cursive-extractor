import json
import os
from pdf2image import convert_from_path
from PIL import Image, ImageDraw

# Configuration
PDF_PATH = "data/content/test_document.pdf"
JSON_PATH = "data/content/handwriting_dataset_precise.json"
OUTPUT_IMAGE_PATH = "data/content/overlay_visualized.jpg"
TARGET_PAGE_INDEX = 1  # Page 2

def generate_overlay():
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
    
    print(f"✅ Loaded {len(data)} entries.")

    # 3. Draw Strokes
    draw = ImageDraw.Draw(page_image)
    
    # Use green for the overlay as requested
    STROKE_COLOR = (0, 255, 0) 
    STROKE_WIDTH = 2

    for entry in data:
        points = entry.get("points", [])
        if not points:
            continue
        
        # Points are [x, y, p] normalized to 0-1
        # We need to scale them to image dimensions
        
        # Convert to pixel coordinates
        pixel_points = []
        for pt in points:
            x_norm, y_norm, p = pt
            x = x_norm * width
            y = y_norm * height
            pixel_points.append((x, y, p))
        
        # Draw the stroke
        # We need to handle pen up/down (p=1 is down, p=0 is up?)
        # In the React code: 
        # if (point[2] === 1) { ctx.lineTo... } else { ctx.moveTo... }
        
        # Since we are drawing on a static image with PIL, we can just draw lines between consecutive points where p=1
        # But wait, the React code does:
        # stroke.forEach(point => { if (point[2] === 1) lineTo else moveTo })
        # This implies a single "stroke" array in React might contain multiple actual strokes separated by p=0?
        # In the JSON, "points" is a flat list.
        
        # Let's iterate and draw lines
        if not pixel_points:
            continue
            
        # Start at the first point
        start_x, start_y, start_p = pixel_points[0]
        
        for i in range(1, len(pixel_points)):
            end_x, end_y, end_p = pixel_points[i]
            
            # If the current point has p=1, we draw a line to it from the previous point
            # But we only draw if the *previous* point was also part of the stroke?
            # The React code:
            # ctx.moveTo(stroke[0][0], stroke[0][1]);
            # stroke.forEach(point => { if (point[2] === 1) ctx.lineTo... else ctx.moveTo... })
            
            # So if p=1, we draw line from previous position to current.
            # If p=0, we move to current position (without drawing).
            
            if end_p == 1:
                # Draw line from previous point to current point
                # But wait, if the previous point was a "move", we shouldn't draw *from* it?
                # Actually, in React `lineTo` draws from the *current path position*.
                # `moveTo` updates the current path position.
                
                # So:
                # 1. Start: Move to p[0]
                # 2. Loop p[1..n]:
                #    If p[i].p == 1: Draw line from p[i-1] to p[i]
                #    Else: Move to p[i]
                
                # However, if p[i-1] was a "lift" (p=0), and p[i] is "down" (p=1), 
                # we should draw from p[i-1] to p[i]?
                # The React code says: `if (point[2] === 1) { ctx.lineTo(point[0], point[1]); } else { ctx.moveTo(point[0], point[1]); }`
                # So if p=1, it draws a line *from the last position* to the new position.
                # The last position is whatever the previous command set it to.
                
                draw.line([(pixel_points[i-1][0], pixel_points[i-1][1]), (end_x, end_y)], fill=STROKE_COLOR, width=STROKE_WIDTH)
            else:
                # p=0, effectively a move (though PIL doesn't have state like canvas context, we just don't draw)
                pass

    # 4. Save Image
    page_image.save(OUTPUT_IMAGE_PATH)
    print(f"🖼️ Overlay saved to: {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    generate_overlay()
