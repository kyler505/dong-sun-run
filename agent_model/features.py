"""
Feature extraction for DQN input.
Converts game state into a fixed-size feature vector.
"""

from .heuristics import (
    get_safe_moves,
    flood_fill_area,
    get_distance_to_nearest_obstacle,
    get_opponent_head_position,
    manhattan_distance,
    apply_direction,
)
from case_closed_game import Direction


def extract_features(state, player_number, direction, board_width=20, board_height=18):
    """
    Extract feature vector for a given state and potential move direction.

    Features (16 total):
    1. Distance to obstacle in move direction (normalized)
    2. Distance to obstacle UP (normalized)
    3. Distance to obstacle DOWN (normalized)
    4. Distance to obstacle LEFT (normalized)
    5. Distance to obstacle RIGHT (normalized)
    6. Flood-fill area from move position (normalized)
    7. Flood-fill area from current position (normalized)
    8. Distance to opponent head (normalized)
    9. My trail length (normalized)
    10. Opponent trail length (normalized)
    11. My boosts remaining (normalized)
    12. Opponent boosts remaining (normalized)
    13. Turn count (normalized)
    14. Is move safe (0 or 1)
    15. Trail length advantage (my_length - opponent_length, normalized)
    16. Space advantage (my_area - opponent_area, normalized)

    Args:
        state: Game state dict
        player_number: 1 or 2
        direction: Direction enum to evaluate
        board_width: Width of board
        board_height: Height of board

    Returns:
        List of 16 float values
    """
    # Extract my info and opponent info
    if player_number == 1:
        my_trail = state.get('agent1_trail', [])
        other_trail = state.get('agent2_trail', [])
        my_length = state.get('agent1_length', 0)
        other_length = state.get('agent2_length', 0)
        my_boosts = state.get('agent1_boosts', 3)
        other_boosts = state.get('agent2_boosts', 3)
        my_alive = state.get('agent1_alive', True)
    else:
        my_trail = state.get('agent2_trail', [])
        other_trail = state.get('agent1_trail', [])
        my_length = state.get('agent2_length', 0)
        other_length = state.get('agent1_length', 0)
        my_boosts = state.get('agent2_boosts', 3)
        other_boosts = state.get('agent1_boosts', 3)
        my_alive = state.get('agent2_alive', True)

    turn_count = state.get('turn_count', 0)

    if not my_alive or not my_trail:
        # Return zeros if dead or no trail
        return [0.0] * 16

    # Convert list positions to tuples
    my_trail = [tuple(p) if isinstance(p, list) else p for p in my_trail]
    other_trail = [tuple(p) if isinstance(p, list) else p for p in other_trail]

    head = my_trail[-1]    # Calculate new position after move
    new_pos = apply_direction(head, direction, board_width, board_height)

    # Get distance to obstacles in all directions from current head
    obstacle_distances = get_distance_to_nearest_obstacle(head, my_trail, other_trail, board_width, board_height)

    # Get distance in move direction
    move_direction_distance = obstacle_distances.get(direction, 1)

    # Check if move is safe
    safe_moves = get_safe_moves(state, player_number, board_width, board_height)
    safe_move_strs = [move_str for _, move_str in safe_moves]
    direction_to_str = {
        Direction.UP: "UP",
        Direction.DOWN: "DOWN",
        Direction.LEFT: "LEFT",
        Direction.RIGHT: "RIGHT",
    }
    is_safe = 1.0 if direction_to_str.get(direction, "") in safe_move_strs else 0.0

    # Flood-fill from new position (if safe)
    if is_safe:
        # Simulate the move by adding new_pos to my trail temporarily
        simulated_my_trail = list(my_trail) + [new_pos]
        flood_area_after_move = flood_fill_area(new_pos, simulated_my_trail, other_trail, board_width, board_height)
    else:
        flood_area_after_move = 0

    # Flood-fill from current position
    flood_area_current = flood_fill_area(head, my_trail, other_trail, board_width, board_height)

    # Distance to opponent
    opponent_head = get_opponent_head_position(state, player_number)
    # Convert to tuple if it's a list
    if opponent_head and isinstance(opponent_head, list):
        opponent_head = tuple(opponent_head)
    dist_to_opponent = manhattan_distance(head, opponent_head, board_width, board_height)

    # Calculate opponent's flood-fill area
    if opponent_head:
        opponent_flood_area = flood_fill_area(opponent_head, other_trail, my_trail, board_width, board_height)
    else:
        opponent_flood_area = 0

    # Normalization constants
    max_distance = board_width + board_height
    max_area = board_width * board_height
    max_turns = 500

    # Build feature vector
    features = [
        move_direction_distance / max_distance,  # 1
        obstacle_distances.get(Direction.UP, 1) / max_distance,  # 2
        obstacle_distances.get(Direction.DOWN, 1) / max_distance,  # 3
        obstacle_distances.get(Direction.LEFT, 1) / max_distance,  # 4
        obstacle_distances.get(Direction.RIGHT, 1) / max_distance,  # 5
        flood_area_after_move / max_area,  # 6
        flood_area_current / max_area,  # 7
        min(dist_to_opponent / max_distance, 1.0),  # 8
        my_length / max_turns,  # 9
        other_length / max_turns,  # 10
        my_boosts / 3.0,  # 11
        other_boosts / 3.0,  # 12
        turn_count / max_turns,  # 13
        is_safe,  # 14
        (my_length - other_length) / max_turns,  # 15 (can be negative)
        (flood_area_current - opponent_flood_area) / max_area,  # 16 (can be negative)
    ]

    return features


def extract_all_direction_features(state, player_number, board_width=20, board_height=18):
    """
    Extract features for all 4 directions.

    Args:
        state: Game state dict
        player_number: 1 or 2
        board_width: Width of board
        board_height: Height of board

    Returns:
        Dict mapping Direction enum to feature list
    """
    directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

    features_dict = {}
    for direction in directions:
        features_dict[direction] = extract_features(state, player_number, direction, board_width, board_height)

    return features_dict
