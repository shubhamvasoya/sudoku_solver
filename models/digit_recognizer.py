import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2


class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)  # 10 classes: 0-9

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)


class DigitRecognizer:
    def __init__(self, model_path='data/mnist_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DigitCNN().to(self.device)
        self.model_path = model_path

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.load_model()

    def load_model(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            print("Model loaded successfully!")
        except FileNotFoundError:
            print(f"Model not found at {self.model_path}. Please train the model first.")

    def predict_digit(self, image):
        """Enhanced digit prediction with confidence scoring"""
        if image is None or image.size == 0:
            return 0

        # Convert to PIL Image if it's numpy array
        if isinstance(image, np.ndarray):
            # Ensure it's grayscale
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Enhanced preprocessing for better recognition
            image = self.enhanced_preprocess(image)
            image = Image.fromarray(image)

        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.exp(output)
            confidence, prediction = torch.max(probabilities, 1)

            # Higher confidence threshold and additional validation
            if confidence.item() > 0.85 and self.validate_prediction(image, prediction.item()):
                return prediction.item()
            else:
                return 0

    def enhanced_preprocess(self, image):
        """Advanced preprocessing to improve digit recognition"""
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(image, 9, 75, 75)

        # Use multiple thresholding approaches and combine
        # Method 1: Adaptive threshold
        thresh1 = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)

        # Method 2: Otsu's thresholding
        _, thresh2 = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Combine both methods
        combined = cv2.bitwise_and(thresh1, thresh2)

        # Invert if needed (digits should be white on black)
        if np.mean(combined) > 127:
            combined = 255 - combined

        # Remove small noise
        kernel = np.ones((2, 2), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        # Fill small holes
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        return combined

    def validate_prediction(self, image, prediction):
        """Additional validation to reduce false positives"""
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Check if image has enough content to be a digit
        white_pixels = np.count_nonzero(image)
        total_pixels = image.shape[0] * image.shape[1]

        # If too few pixels, likely empty
        if white_pixels < 50:
            return False

        # If too many pixels, likely noise or grid lines
        if white_pixels > (total_pixels * 0.8):
            return False

        # Check aspect ratio and connectivity
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False

        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Check if contour area is reasonable
        if cv2.contourArea(largest_contour) < 30:
            return False

        # Check bounding box aspect ratio
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h

        # Digits should have reasonable aspect ratios
        if aspect_ratio > 2.0 or aspect_ratio < 0.3:
            return False

        return True

    def predict_digit_with_confidence(self, image):
        """Predict digit and return confidence score"""
        if image is None or image.size == 0:
            return 0, 0.0

        # Convert to PIL Image if it's numpy array
        if isinstance(image, np.ndarray):
            # Ensure it's grayscale
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Enhanced preprocessing
            image = self.enhanced_preprocess(image)
            image = Image.fromarray(image)

        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.exp(output)
            confidence, prediction = torch.max(probabilities, 1)

            return prediction.item(), confidence.item()

    def predict_grid(self, cell_images):
        """Enhanced grid prediction with multiple validation passes"""
        grid = []
        debug_info = []

        # First pass: Initial predictions
        for i, row in enumerate(cell_images):
            grid_row = []
            debug_row = []

            for j, cell_img in enumerate(row):
                if self.is_empty_cell_advanced(cell_img):
                    grid_row.append(0)
                    debug_row.append((0, 0.0, "empty"))
                else:
                    digit, confidence = self.predict_digit_with_confidence(cell_img)

                    # Multiple confidence checks
                    if confidence > 0.85 and self.validate_prediction(cell_img, digit):
                        grid_row.append(digit)
                        debug_row.append((digit, confidence, "confident"))
                    elif confidence > 0.7:
                        # Secondary validation for medium confidence
                        if self.cross_validate_digit(cell_img, digit):
                            grid_row.append(digit)
                            debug_row.append((digit, confidence, "cross_validated"))
                        else:
                            grid_row.append(0)
                            debug_row.append((digit, confidence, "failed_validation"))
                    else:
                        grid_row.append(0)
                        debug_row.append((digit, confidence, "low_confidence"))

            grid.append(grid_row)
            debug_info.append(debug_row)

        # Post-processing: Remove isolated predictions that might be noise
        grid = self.post_process_grid(grid)

        # Print debug information
        print("\nDigit Recognition Debug Info:")
        for i, debug_row in enumerate(debug_info):
            for j, (digit, conf, status) in enumerate(debug_row):
                if status != "empty":
                    print(f"Cell [{i},{j}]: {digit} (conf: {conf:.3f}, {status})")

        return grid

    def cross_validate_digit(self, image, predicted_digit):
        """Cross-validate digit prediction with multiple approaches"""
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Apply different preprocessing and check consistency
        variations = []

        # Variation 1: Different morphological operations
        kernel = np.ones((3, 3), np.uint8)
        var1 = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        var1_tensor = self.transform(Image.fromarray(var1)).unsqueeze(0).to(self.device)

        # Variation 2: Slight dilation
        var2 = cv2.dilate(image, kernel, iterations=1)
        var2_tensor = self.transform(Image.fromarray(var2)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Test variations
            out1 = self.model(var1_tensor)
            out2 = self.model(var2_tensor)

            pred1 = torch.argmax(out1, dim=1).item()
            pred2 = torch.argmax(out2, dim=1).item()

            # If majority agrees with original prediction, accept it
            predictions = [predicted_digit, pred1, pred2]
            if predictions.count(predicted_digit) >= 2:
                return True

        return False

    def post_process_grid(self, grid):
        """Remove isolated predictions that are likely noise"""
        processed = [row[:] for row in grid]

        for i in range(9):
            for j in range(9):
                if processed[i][j] != 0:
                    # Count neighboring non-empty cells
                    neighbors = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < 9 and 0 <= nj < 9 and grid[ni][nj] != 0:
                                neighbors += 1

                    # If completely isolated, treat as empty
                    if neighbors == 0:
                        processed[i][j] = 0

        return processed

    def is_empty_cell_advanced(self, image, threshold=0.02):
        """Advanced empty cell detection with multiple checks"""
        if image is None:
            return True

        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Multiple threshold checks
            # Check 1: Pixel density
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            white_pixels = np.count_nonzero(binary)
            total_pixels = image.shape[0] * image.shape[1]

            if (white_pixels / total_pixels) < threshold:
                return True

            # Check 2: Connected components analysis
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

            if num_labels <= 1:  # Only background
                return True

            # Check 3: Largest component size
            if num_labels > 1:
                largest_area = max(stats[1:, cv2.CC_STAT_AREA])
                if largest_area < 30:  # Too small to be a digit
                    return True

            # Check 4: Aspect ratio of largest component
            if num_labels > 1:
                largest_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                w = stats[largest_idx, cv2.CC_STAT_WIDTH]
                h = stats[largest_idx, cv2.CC_STAT_HEIGHT]

                if w > 0 and h > 0:
                    aspect_ratio = w / h
                    # Reject if aspect ratio is too extreme
                    if aspect_ratio > 3.0 or aspect_ratio < 0.2:
                        return True

            # Check 5: Variance test (uniform regions are likely empty)
            if np.var(image) < 100:  # Very uniform, likely empty
                return True

            return False

        return True

    def is_empty_cell(self, image, threshold=0.03):
        """Enhanced empty cell detection - kept for backward compatibility"""
        return self.is_empty_cell_advanced(image, threshold)
