class SudokuSolver:
    def __init__(self):
        pass

    def is_valid(self, grid, row, col, num):
        """Check if placing num at (row, col) is valid"""
        # Check row
        for j in range(9):
            if grid[row][j] == num:
                return False

        # Check column
        for i in range(9):
            if grid[i][col] == num:
                return False

        # Check 3x3 box
        start_row = (row // 3) * 3
        start_col = (col // 3) * 3

        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if grid[i][j] == num:
                    return False

        return True

    def solve(self, grid):
        """Solve Sudoku using backtracking"""
        # Create a copy to avoid modifying original
        solution = [row[:] for row in grid]

        if self._solve_helper(solution):
            return solution
        return None

    def _solve_helper(self, grid):
        """Recursive helper for solving"""
        for i in range(9):
            for j in range(9):
                if grid[i][j] == 0:
                    for num in range(1, 10):
                        if self.is_valid(grid, i, j, num):
                            grid[i][j] = num

                            if self._solve_helper(grid):
                                return True

                            grid[i][j] = 0  # Backtrack

                    return False
        return True

    def is_valid_sudoku(self, grid):
        """Check if the current grid state is valid"""
        for i in range(9):
            for j in range(9):
                if grid[i][j] != 0:
                    num = grid[i][j]
                    grid[i][j] = 0  # Temporarily remove

                    if not self.is_valid(grid, i, j, num):
                        grid[i][j] = num  # Restore
                        return False

                    grid[i][j] = num  # Restore
        return True