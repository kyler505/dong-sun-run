"""
Heuristic-based safety and survival logic for Case Closed agent.
Includes flood-fill, safe move detection, and strategic filtering.
"""

from collections import deque
from case_closed_game import Direction


def get_safe_moves(state, player_number, board_width=20, board_height=18):
    """
    Get all moves that don't immediately result in death.

    Args:
        state: Game state dict with trails, board, etc.
        player_number: 1 or 2
        board_width: Width of the game board
        board_height: Height of the game board

    Returns:
        List of (Direction, move_string) tuples that are safe
    """
    if player_number == 1:
        my_trail = state.get('agent1_trail', [])
        other_trail = state.get('agent2_trail', [])
        my_alive = state.get('agent1_alive', True)
    else:
        my_trail = state.get('agent2_trail', [])
        other_trail = state.get('agent1_trail', [])
        my_alive = state.get('agent2_alive', True)

    if not my_alive or not my_trail:
        return [(Direction.RIGHT, "RIGHT")]

    # Convert list positions to tuples
    my_trail = [tuple(p) if isinstance(p, list) else p for p in my_trail]
    other_trail = [tuple(p) if isinstance(p, list) else p for p in other_trail]

    head = my_trail[-1]

    # Get current direction
    current_dir = get_current_direction(my_trail)    # All possible moves
    all_moves = [
        (Direction.UP, "UP"),
        (Direction.DOWN, "DOWN"),
        (Direction.LEFT, "LEFT"),
        (Direction.RIGHT, "RIGHT"),
    ]

    # Filter out opposite direction (invalid)
    opposite_map = {
        Direction.UP: Direction.DOWN,
        Direction.DOWN: Direction.UP,
        Direction.LEFT: Direction.RIGHT,
        Direction.RIGHT: Direction.LEFT,
    }

    safe_moves = []

    for direction, move_str in all_moves:
        # Skip opposite direction
        if direction == opposite_map.get(current_dir):
            continue

        # Calculate new position
        new_pos = apply_direction(head, direction, board_width, board_height)

        # Check if position is safe (not in any trail)
        if is_position_safe(new_pos, my_trail, other_trail):
            safe_moves.append((direction, move_str))

    # If no safe moves, return all non-opposite moves (let game handle death)
    if not safe_moves:
        for direction, move_str in all_moves:
            if direction != opposite_map.get(current_dir):
                safe_moves.append((direction, move_str))

    return safe_moves


def get_current_direction(trail):
    """
    Determine current direction from trail.

    Args:
        trail: List of (x, y) positions

    Returns:
        Direction enum
    """
    if len(trail) < 2:
        return Direction.RIGHT

    head = trail[-1]
    prev = trail[-2]

    dx = head[0] - prev[0]
    dy = head[1] - prev[1]

    # Normalize for torus wrapping
    if abs(dx) > 1:
        dx = -1 if dx > 0 else 1
    if abs(dy) > 1:
        dy = -1 if dy > 0 else 1

    if dx == 1:
        return Direction.RIGHT
    elif dx == -1:
        return Direction.LEFT
    elif dy == 1:
        return Direction.DOWN
    elif dy == -1:
        return Direction.UP
    else:
        return Direction.RIGHT


def apply_direction(pos, direction, board_width=20, board_height=18):
    """
    Apply a direction to a position with torus wrapping.

    Args:
        pos: (x, y) tuple
        direction: Direction enum
        board_width: Width of board
        board_height: Height of board

    Returns:
        New (x, y) tuple after applying direction
    """
    x, y = pos
    dx, dy = direction.value
    new_x = (x + dx) % board_width
    new_y = (y + dy) % board_height
    return (new_x, new_y)


def is_position_safe(pos, my_trail, other_trail):
    """
    Check if a position is not occupied by any trail.

    Args:
        pos: (x, y) tuple to check
        my_trail: My trail list
        other_trail: Opponent's trail list

    Returns:
        True if safe, False otherwise
    """
    # Convert trails to sets for O(1) lookup
    # Handle both list and tuple positions
    my_trail_set = set(tuple(p) if isinstance(p, list) else p for p in my_trail) if my_trail else set()
    other_trail_set = set(tuple(p) if isinstance(p, list) else p for p in other_trail) if other_trail else set()

    return pos not in my_trail_set and pos not in other_trail_set
def flood_fill_area(start_pos, my_trail, other_trail, board_width=20, board_height=18, max_depth=50):
    """
    Calculate the reachable area from a starting position using BFS.

    Args:
        start_pos: (x, y) starting position
        my_trail: My trail list
        other_trail: Opponent's trail list
        board_width: Width of board
        board_height: Height of board
        max_depth: Maximum BFS depth (for performance)

    Returns:
        Number of reachable empty cells
    """
    # Convert trails to tuples if they're lists
    occupied = set(tuple(p) if isinstance(p, list) else p for p in my_trail) | \
               set(tuple(p) if isinstance(p, list) else p for p in other_trail)

    visited = set()
    queue = deque([(start_pos, 0)])
    visited.add(start_pos)

    reachable = 0

    while queue:
        pos, depth = queue.popleft()

        if depth > max_depth:
            continue

        reachable += 1

        # Explore all 4 directions
        for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
            new_pos = apply_direction(pos, direction, board_width, board_height)

            if new_pos not in visited and new_pos not in occupied:
                visited.add(new_pos)
                queue.append((new_pos, depth + 1))

    return reachable
def get_distance_to_nearest_obstacle(pos, my_trail, other_trail, board_width=20, board_height=18):
    """
    Get Manhattan distance to nearest trail cell in each direction.

    Args:
        pos: (x, y) current position
        my_trail: My trail list
        other_trail: Opponent's trail list
        board_width: Width of board
        board_height: Height of board

    Returns:
        Dict with keys UP, DOWN, LEFT, RIGHT and distance values
    """
    # Convert trails to tuples if they're lists
    occupied = set(tuple(p) if isinstance(p, list) else p for p in my_trail) | \
               set(tuple(p) if isinstance(p, list) else p for p in other_trail)

    distances = {}

    for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
        dist = 1
        current_pos = pos

        # Search up to half the board dimension
        max_search = max(board_width, board_height) // 2

        while dist <= max_search:
            current_pos = apply_direction(current_pos, direction, board_width, board_height)

            if current_pos in occupied:
                break

            dist += 1

        distances[direction] = min(dist, max_search)

    return distances
def get_opponent_head_position(state, player_number):
    """
    Get opponent's head position.

    Args:
        state: Game state dict
        player_number: My player number (1 or 2)

    Returns:
        (x, y) tuple or None
    """
    if player_number == 1:
        other_trail = state.get('agent2_trail', [])
    else:
        other_trail = state.get('agent1_trail', [])

    if other_trail:
        return other_trail[-1]
    return None


def manhattan_distance(pos1, pos2, board_width=20, board_height=18):
    """
    Calculate Manhattan distance with torus wrapping.

    Args:
        pos1: (x, y) tuple
        pos2: (x, y) tuple
        board_width: Width of board
        board_height: Height of board

    Returns:
        Minimum Manhattan distance considering wraparound
    """
    if pos1 is None or pos2 is None:
        return float('inf')

    dx = abs(pos1[0] - pos2[0])
    dy = abs(pos1[1] - pos2[1])

    # Consider wraparound
    dx = min(dx, board_width - dx)
    dy = min(dy, board_height - dy)

    return dx + dy
