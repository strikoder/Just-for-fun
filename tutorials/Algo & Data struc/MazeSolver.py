# Base cases: 1. it's a wall, 2. off the map, 3.it's the end, 4. if we have seen it

path = [
"#####E#",
"#$$$$$#",
"#S#####"
]


def find_start(path):
    for i, row in enumerate(path):
        for j, cell in enumerate(row):
            if cell == 'S':
                return i, j
    return None

def can_move(path, visited, row, col):
    # Check if the position is within bounds and not a wall or visited
    return (0 <= row < len(path) and 0 <= col < len(path[0]) and
            path[row][col] != '#' and not visited[row][col])

def find_path(path, row, col, visited):
    # If the position is outside the bounds or is a wall, return False
    if not can_move(path, visited, row, col):
        return False
    
    # Mark the cell as visited
    visited[row][col] = True

    # If the current cell is the end, return True
    if path[row][col] == 'E':
        return True

    # Explore adjacent cells (up, down, left, right)
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    for dr, dc in directions:
        next_row, next_col = row + dr, col + dc
        if can_move(path, visited, next_row, next_col) and find_path(path, next_row, next_col, visited):
            return True

    # Backtrack: Unmark the visited cell if no path is found (optional here since we return immediately on success)
    # visited[row][col] = False  # Uncomment if you're marking the path or need to backtrack

    return False

# Convert path to a list of lists for easier manipulation
path_list = [list(row) for row in path]

# Initialize a visited matrix
visited = [[False for _ in row] for row in path_list]

# Find the start point
start_row, start_col = find_start(path_list)

# Find and print if a path exists
if find_path(path_list, start_row, start_col, visited):
    print("Path found!")
else:
    print("No path found.")


