import cv2
import numpy as np


def preprocess_image(image):
    """Preprocess image for better grid detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    return thresh


def find_sudoku_grid(image):
    """Find and extract Sudoku grid from image"""
    processed = preprocess_image(image)

    # Find contours
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest rectangular contour
    largest_contour = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if it's roughly rectangular (4 corners)
            if len(approx) == 4 and area > 10000:  # Minimum area threshold
                largest_contour = approx
                max_area = area

    if largest_contour is None:
        return None, None

    # Order points: top-left, top-right, bottom-right, bottom-left
    points = largest_contour.reshape(4, 2)
    ordered_points = order_points(points)

    # Perspective transform
    grid_size = 450  # Output size
    dst_points = np.array([
        [0, 0],
        [grid_size, 0],
        [grid_size, grid_size],
        [0, grid_size]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(ordered_points.astype(np.float32), dst_points)
    warped = cv2.warpPerspective(image, matrix, (grid_size, grid_size))

    return warped, matrix


def order_points(pts):
    """Order points in clockwise order starting from top-left"""
    rect = np.zeros((4, 2), dtype=np.float32)

    # Sum and difference to find corners
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    return rect


def extract_cells(grid_image):
    """Enhanced cell extraction with better preprocessing"""
    cells = []
    cell_size = grid_image.shape[0] // 9

    # Convert to grayscale for processing
    gray = cv2.cvtColor(grid_image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    for i in range(9):
        row_cells = []
        for j in range(9):
            # Extract cell with small margin to avoid grid lines
            margin = 3
            y1 = i * cell_size + margin
            y2 = (i + 1) * cell_size - margin
            x1 = j * cell_size + margin
            x2 = (j + 1) * cell_size - margin

            # Ensure we don't go out of bounds
            y1 = max(0, y1)
            y2 = min(gray.shape[0], y2)
            x1 = max(0, x1)
            x2 = min(gray.shape[1], x2)

            cell = gray[y1:y2, x1:x2]

            # Clean up cell
            cell = preprocess_cell(cell)
            row_cells.append(cell)

        cells.append(row_cells)

    return cells


def preprocess_cell(cell):
    """Enhanced preprocessing for individual cell digit recognition"""
    if cell is None or cell.size == 0:
        return None

    # Apply multiple filters for noise reduction
    # 1. Bilateral filter to preserve edges while reducing noise
    denoised = cv2.bilateralFilter(cell, 9, 75, 75)

    # 2. Gaussian blur for smoothing
    blurred = cv2.GaussianBlur(denoised, (3, 3), 0)

    # 3. Multiple thresholding approaches
    # Adaptive threshold
    thresh1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

    # Otsu's method
    _, thresh2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Combine both thresholding methods
    thresh = cv2.bitwise_and(thresh1, thresh2)

    # 4. Morphological operations to clean up
    kernel = np.ones((2, 2), np.uint8)

    # Remove small noise (opening)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Fill small gaps (closing)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 5. Find and process the main digit contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Filter contours by area and aspect ratio
        valid_contours = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 30:  # Minimum area
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h

                # Valid digit aspect ratios
                if 0.2 < aspect_ratio < 2.5:
                    valid_contours.append(contour)

        if valid_contours:
            # Get the largest valid contour
            largest_contour = max(valid_contours, key=cv2.contourArea)

            # Create a mask for the digit
            mask = np.zeros(thresh.shape, dtype=np.uint8)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)

            # Apply mask to clean image
            thresh = cv2.bitwise_and(thresh, mask)

            # Get tight bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Add padding
            padding = 8
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(cell.shape[1] - x, w + 2 * padding)
            h = min(cell.shape[0] - y, h + 2 * padding)

            # Crop to bounding box
            cropped = thresh[y:y + h, x:x + w]

            # Resize to square with proper padding
            max_dim = max(w, h)
            square_size = max(max_dim, 32)  # Minimum size

            # Create square canvas
            square = np.zeros((square_size, square_size), dtype=np.uint8)

            # Center the digit
            start_x = (square_size - w) // 2
            start_y = (square_size - h) // 2

            square[start_y:start_y + h, start_x:start_x + w] = cropped

            # Final cleanup
            kernel_final = np.ones((1, 1), np.uint8)
            square = cv2.morphologyEx(square, cv2.MORPH_CLOSE, kernel_final)

            return square

    return thresh


def draw_solution(image, original_grid, solution_grid, transform_matrix):
    """Draw solution on the original image"""
    if solution_grid is None:
        return image

    result = image.copy()

    # Grid dimensions
    grid_size = 450
    cell_size = grid_size // 9

    # Create solution overlay
    overlay = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)

    for i in range(9):
        for j in range(9):
            if original_grid[i][j] == 0:  # Only draw solution digits
                digit = solution_grid[i][j]
                if digit != 0:
                    # Calculate position
                    x = j * cell_size + cell_size // 2
                    y = i * cell_size + cell_size // 2

                    # Draw digit
                    cv2.putText(overlay, str(digit), (x - 15, y + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Transform solution back to original perspective
    if transform_matrix is not None:
        inverse_matrix = cv2.invert(transform_matrix)[1]
        h, w = image.shape[:2]
        solution_warped = cv2.warpPerspective(overlay, inverse_matrix, (w, h))

        # Blend with original image
        mask = cv2.cvtColor(solution_warped, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]

        result = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(mask))
        result = cv2.add(result, solution_warped)

    return result