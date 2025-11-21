import cv2
import numpy as np
import sys

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
    
    print("⏳ Starting skeletonization...")
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
    # Or find an endpoint (pixel with 1 neighbor)
    # For now, simple sort by x then y to find a start, then nearest neighbor
    points.sort(key=lambda p: (p[0], p[1]))
    
    ordered_path = []
    current_point = points[0]
    ordered_path.append(current_point)
    points.remove(current_point)
    
    while points:
        # Find nearest neighbor to current_point
        # Since it's a skeleton, the next point should be adjacent (dist <= sqrt(2))
        # We search for points within distance 2
        candidates = [p for p in points if max(abs(p[0]-current_point[0]), abs(p[1]-current_point[1])) <= 1]
        
        if candidates:
            # Pick the first one (DFS like)
            next_point = candidates[0]
            ordered_path.append(next_point)
            points.remove(next_point)
            current_point = next_point
        else:
            # Jump to the next available point (discontinuity / new stroke)
            # In a real vectorization, this would be a new path
            # For this single list, we just jump
            next_point = points[0]
            ordered_path.append(next_point) # Maybe add a separator?
            points.remove(next_point)
            current_point = next_point
            
    return ordered_path

def main():
    input_path = "data/content/temp_crop.jpg"
    output_path = "data/content/vectorized_crop.jpg"
    
    print(f"📖 Reading {input_path}...")
    img = cv2.imread(input_path)
    if img is None:
        print(f"❌ Error: Could not read {input_path}")
        return

    # 1. Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Gaussian Blur to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Thresholding (Otsu)
    # Invert so ink is White (255) and background is Black (0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Optional: Morphological closing to fill small gaps in ink
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 2. Skeletonization
    skeleton = skeletonize(closed)
    print("✅ Skeletonization complete.")
    
    # 3. Vectorization (Path Calculation)
    # Get ordered points
    path_points = order_points(skeleton)
    print(f"✅ Calculated path with {len(path_points)} points.")
    
    # 4. Visualization
    # Create a copy of the original image for overlay
    overlay = img.copy()
    
    # Draw the skeleton pixels
    # We can draw them as a continuous line based on our ordered path
    if len(path_points) > 1:
        for i in range(len(path_points) - 1):
            p1 = path_points[i]
            p2 = path_points[i+1]
            
            # Only draw line if they are close (connected)
            dist = max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))
            if dist <= 2:
                cv2.line(overlay, p1, p2, (0, 255, 0), 2) # Green line, thickness 2
            else:
                # If jump, just draw the point
                cv2.circle(overlay, p2, 1, (0, 255, 0), -1)
    
    # Also draw the raw skeleton pixels in red just to be sure we didn't miss anything in the path
    # (Optional, but let's stick to the green line as requested)
    
    cv2.imwrite(output_path, overlay)
    print(f"🖼️ Saved visualization to {output_path}")

if __name__ == "__main__":
    main()
