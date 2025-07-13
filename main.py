import cv2
import numpy as np
import os
from models.digit_recognizer import DigitRecognizer
from models.sudoku_solver import SudokuSolver
from utils.image_processing import (
    find_sudoku_grid, extract_cells, draw_solution, preprocess_image
)


class SudokuApp:
    def __init__(self):
        self.digit_recognizer = DigitRecognizer()
        self.sudoku_solver = SudokuSolver()
        self.current_grid = None
        self.solution_grid = None
        self.transform_matrix = None

    def run(self):
        # Initialize camera
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print("AI Sudoku Solver Started!")
        print("Controls:")
        print("  SPACE - Capture and solve Sudoku")
        print("  's' - Save current cells for debugging")
        print("  'r' - Reset current solution")
        print("  'q' - Quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Display frame
            display_frame = frame.copy()

            # Add instructions
            cv2.putText(display_frame, "Press SPACE to solve Sudoku",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press 'r' to reset, 'q' to quit",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # If we have a solution, draw it
            if self.solution_grid is not None and self.current_grid is not None:
                display_frame = draw_solution(display_frame, self.current_grid,
                                              self.solution_grid, self.transform_matrix)

            cv2.imshow('Sudoku Solver', display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):  # Space key
                self.process_frame(frame)
            elif key == ord('s'):  # Save cells for debugging
                self.save_debug_cells(frame)
            elif key == ord('r'):  # Reset
                self.reset_solution()
            elif key == ord('q'):  # Quit
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        """Enhanced frame processing with manual correction option"""
        print("Processing frame...")

        # Find Sudoku grid
        grid_image, transform_matrix = find_sudoku_grid(frame)

        if grid_image is None:
            print("No Sudoku grid detected!")
            return

        print("Grid detected! Extracting cells...")

        # Extract cells
        cells = extract_cells(grid_image)

        # Recognize digits
        print("Recognizing digits...")
        detected_grid = self.digit_recognizer.predict_grid(cells)

        # Display detected grid
        self.print_grid(detected_grid, "Detected Grid:")

        # Ask user if they want to correct any digits
        print("\nIf any digits are wrong, you can correct them.")
        print("Enter corrections in format: row,col,digit (e.g., 0,1,5)")
        print("Press Enter to skip corrections")

        while True:
            correction = input("Enter correction (or press Enter to continue): ").strip()
            if not correction:
                break

            try:
                parts = correction.split(',')
                if len(parts) == 3:
                    row, col, digit = map(int, parts)
                    if 0 <= row < 9 and 0 <= col < 9 and 0 <= digit <= 9:
                        detected_grid[row][col] = digit
                        print(f"Corrected cell [{row},{col}] to {digit}")
                        self.print_grid(detected_grid, "Corrected Grid:")
                    else:
                        print("Invalid values. Use row,col,digit with values 0-8 for row/col and 0-9 for digit")
                else:
                    print("Invalid format. Use: row,col,digit")
            except ValueError:
                print("Invalid input. Use numbers only in format: row,col,digit")

        # Solve Sudoku
        print("Solving Sudoku...")
        solution = self.sudoku_solver.solve(detected_grid)

        if solution:
            print("Sudoku solved!")
            self.print_grid(solution, "Solution:")

            self.current_grid = detected_grid
            self.solution_grid = solution
            self.transform_matrix = transform_matrix

            # Show solution in separate window
            self.show_solution_window(detected_grid, solution)
        else:
            print("Could not solve Sudoku. Please check if the grid is valid.")
            print("The detected grid might have errors. Try correcting the digits.")

            # Show what makes the grid invalid
            if not self.sudoku_solver.is_valid_sudoku(detected_grid):
                print("The detected grid contains invalid numbers (duplicates in row/column/box).")

    def reset_solution(self):
        """Reset current solution"""
        self.current_grid = None
        self.solution_grid = None
        self.transform_matrix = None
        cv2.destroyWindow('Sudoku Solution')
        print("Solution reset!")

    def save_debug_cells(self, frame):
        """Save individual cells for debugging digit recognition"""
        print("Saving cells for debugging...")

        # Find Sudoku grid
        grid_image, _ = find_sudoku_grid(frame)

        if grid_image is None:
            print("No Sudoku grid detected!")
            return

        # Extract cells
        cells = extract_cells(grid_image)

        # Create debug directory
        import os
        debug_dir = "debug_cells"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)

        # Save each cell
        for i, row in enumerate(cells):
            for j, cell in enumerate(row):
                if cell is not None:
                    filename = f"{debug_dir}/cell_{i}_{j}.png"
                    cv2.imwrite(filename, cell)

        print(f"Saved {9 * 9} cells to {debug_dir}/ directory")
        print("You can examine these images to see why digit recognition might be failing")

    def print_grid(self, grid, title="Grid:"):
        """Print grid to console"""
        print(f"\n{title}")
        for i, row in enumerate(grid):
            if i % 3 == 0 and i != 0:
                print("------+-------+------")

            row_str = ""
            for j, cell in enumerate(row):
                if j % 3 == 0 and j != 0:
                    row_str += "| "
                row_str += str(cell if cell != 0 else '.') + " "

            print(row_str)

    def show_solution_window(self, original_grid, solution_grid):
        """Show solution in a separate window"""
        # Create solution image
        solution_img = np.ones((450, 450, 3), dtype=np.uint8) * 255

        cell_size = 50

        # Draw grid lines
        for i in range(10):
            thickness = 3 if i % 3 == 0 else 1
            cv2.line(solution_img, (i * cell_size, 0),
                     (i * cell_size, 450), (0, 0, 0), thickness)
            cv2.line(solution_img, (0, i * cell_size),
                     (450, i * cell_size), (0, 0, 0), thickness)

        # Draw numbers
        for i in range(9):
            for j in range(9):
                digit = solution_grid[i][j]
                if digit != 0:
                    x = j * cell_size + cell_size // 2
                    y = i * cell_size + cell_size // 2

                    # Color: blue for original, green for solution
                    color = (255, 0, 0) if original_grid[i][j] != 0 else (0, 150, 0)

                    cv2.putText(solution_img, str(digit), (x - 10, y + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('Sudoku Solution', solution_img)


def main():
    # Check if model exists
    if not os.path.exists('data/mnist_model.pth'):
        print("Model not found! Please run train_digit_model.py first.")
        print("Running training now...")

        # Import and run training
        from train_digit_model import train_model
        train_model()

    # Create and run the app
    app = SudokuApp()
    app.run()


if __name__ == "__main__":
    main()