"""
Case Closed Agent with Hybrid Heuristic + DQN + Online Learning

This agent combines:
1. Rule-based heuristics for safety filtering and strategic scoring
2. Deep Q-Network (DQN) for learned decision-making
3. Online learning for real-time adaptation during matches

Online Learning Configuration:
- Conservative learning rate (5e-5) to prevent catastrophic forgetting
- Small replay buffer (500 experiences) for recent game patterns
- Gated updates: minimum buffer size, limited gradient steps per turn
- Gradient clipping (0.5) for stability
- Periodic checkpoint saves to persist improvements across games

To disable online learning, set ONLINE_LEARNING_ENABLED = False below.
"""

import os
import sys
from collections import deque
from threading import Lock

# Ensure imports work from any working directory
# This makes the agent portable when exported to another machine
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from flask import Flask, request, jsonify

from case_closed_game import Game, Direction, GameResult
from agent_model.heuristics import (
    apply_direction,
    flood_fill_area,
    get_current_direction,
    get_safe_moves,
    is_position_safe,
)

# Try to load hybrid DQN components (torch + feature extractor + model loader)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from collections import namedtuple
    from agent_model.dqn_model import load_model, save_model
    from agent_model.features import extract_all_direction_features, extract_features

    DQN_AVAILABLE = True
except ImportError as e:
    torch = None
    load_model = None
    save_model = None
    extract_all_direction_features = None
    extract_features = None
    print(f"⚠️  DQN components not available: {e}")
    print("⚠️  Falling back to heuristic-only agent. Install torch to use DQN.")
    DQN_AVAILABLE = False

# Experience tuple for online learning
if DQN_AVAILABLE:
    Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()

PARTICIPANT = "ParticipantX"
AGENT_NAME = "AgentX"

DEFAULT_BOARD_WIDTH = 20
DEFAULT_BOARD_HEIGHT = 18

DIRECTION_TO_STR = {
    Direction.UP: "UP",
    Direction.DOWN: "DOWN",
    Direction.LEFT: "LEFT",
    Direction.RIGHT: "RIGHT",
}

STR_TO_DIRECTION = {name: direction for direction, name in DIRECTION_TO_STR.items()}

DIRECTION_TO_INDEX = {
    Direction.UP: 0,
    Direction.DOWN: 1,
    Direction.LEFT: 2,
    Direction.RIGHT: 3,
}

INDEX_TO_DIRECTION = {v: k for k, v in DIRECTION_TO_INDEX.items()}

# Online learning components
ONLINE_LEARNING_ENABLED = False  # Disabled - online learning too weak vs offline training
ONLINE_REPLAY_BUFFER = deque(maxlen=500)  # Small ring buffer for recent experiences
ONLINE_OPTIMIZER = None
ONLINE_LEARNING_RATE = 5e-5  # Conservative LR for stability
ONLINE_BATCH_SIZE = 32
ONLINE_MIN_BUFFER_SIZE = 64  # Don't train until we have enough data
ONLINE_GRADIENT_STEPS_PER_TURN = 2  # Max gradient steps per move
ONLINE_UPDATE_COUNTER = 0  # Track number of online updates
LAST_STATE_FEATURES = None  # For capturing transitions
LAST_ACTION = None
LAST_DIRECTION = None

# Game statistics for meta-learning
GAME_STATS = {
    'games_played': 0,
    'total_rewards': [],
    'opponent_encounters': {},  # Track opponents we've faced
}

# Load DQN model if available
DQN_MODEL = None
if DQN_AVAILABLE:
    try:
        # Use path relative to script directory for portability
        model_path = os.path.join(SCRIPT_DIR, 'agent_model', 'models', 'dqn_agent.pth')
        DQN_MODEL = load_model(model_path, input_size=16, hidden_size=64, device='cpu')

        # Load optimizer state for online learning if available
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
                ONLINE_OPTIMIZER = optim.Adam(DQN_MODEL.parameters(), lr=ONLINE_LEARNING_RATE)
                ONLINE_OPTIMIZER.load_state_dict(checkpoint['optimizer_state_dict'])
                # Adjust learning rate to online rate (might differ from training)
                for param_group in ONLINE_OPTIMIZER.param_groups:
                    param_group['lr'] = ONLINE_LEARNING_RATE
                print(f"✅ Optimizer state loaded - online learning enabled")
            else:
                # No optimizer state, create fresh optimizer
                ONLINE_OPTIMIZER = optim.Adam(DQN_MODEL.parameters(), lr=ONLINE_LEARNING_RATE)
                print(f"⚠️  No optimizer state in checkpoint - using fresh optimizer")
        except Exception as e:
            # Create fresh optimizer as fallback
            ONLINE_OPTIMIZER = optim.Adam(DQN_MODEL.parameters(), lr=ONLINE_LEARNING_RATE)
            print(f"⚠️  Could not load optimizer state: {e} - using fresh optimizer")

        # Switch model to training mode for online learning
        DQN_MODEL.train()
        print("✅ DQN model loaded successfully - using Hybrid Heuristic + DQN agent with online learning")

    except Exception as e:
        print(f"⚠️  Could not load DQN model: {e}")
        print("⚠️  Falling back to heuristic-only agent")
else:
    print("⚠️  Using rule-based fallback agent (torch not installed)")


@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity.

    Returns participant and agent_name (so Judge.check_latency can create Agent objects).
    """
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge.

    The judge posts a dictionary with keys matching the Judge.send_state payload
    (board, agent1_trail, agent2_trail, agent1_length, agent2_length, agent1_alive,
    agent2_alive, agent1_boosts, agent2_boosts, turn_count).
    """
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"])
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"])
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])


def _get_trails(state, player_number):
    """Return (my_trail, other_trail) as lists of tuples."""
    if player_number == 1:
        my_trail = state.get('agent1_trail', [])
        other_trail = state.get('agent2_trail', [])
    else:
        my_trail = state.get('agent2_trail', [])
        other_trail = state.get('agent1_trail', [])

    my_trail = [tuple(p) if not isinstance(p, tuple) else p for p in my_trail]
    other_trail = [tuple(p) if not isinstance(p, tuple) else p for p in other_trail]
    return my_trail, other_trail


def _get_board_dimensions(state):
    """Infer board dimensions from state, falling back to defaults."""
    board = state.get("board")
    if isinstance(board, list) and board:
        height = len(board)
        width = len(board[0]) if isinstance(board[0], list) and board[0] else DEFAULT_BOARD_WIDTH
        return width, height
    return DEFAULT_BOARD_WIDTH, DEFAULT_BOARD_HEIGHT


def compute_online_reward(state, next_state, action, player_number, game_over=False):
    """
    Compute reward for online learning based on immediate state changes.
    Mirrors the reward logic from DQNTrainer but adapted for runtime.

    Args:
        state: Current game state dict
        next_state: Next game state dict after action
        action: Action taken (Direction)
        player_number: 1 or 2
        game_over: Whether the game ended

    Returns:
        float reward
    """
    # Default neutral reward
    reward = 0.0

    if game_over:
        # Check who won
        if player_number == 1:
            alive = next_state.get('agent1_alive', True)
            opp_alive = next_state.get('agent2_alive', True)
        else:
            alive = next_state.get('agent2_alive', True)
            opp_alive = next_state.get('agent1_alive', True)

        if not alive and not opp_alive:
            reward = 0.0  # Draw
        elif not alive:
            reward = -10.0  # Loss
        elif not opp_alive:
            reward = 10.0  # Win
        return reward

    # Survival reward (small positive for each turn survived)
    reward += 0.1

    # Space control reward
    try:
        from agent_model.heuristics import flood_fill_area

        if player_number == 1:
            my_trail = [tuple(p) for p in state.get('agent1_trail', [])]
            opp_trail = [tuple(p) for p in state.get('agent2_trail', [])]
            next_my_trail = [tuple(p) for p in next_state.get('agent1_trail', [])]
            next_opp_trail = [tuple(p) for p in next_state.get('agent2_trail', [])]
        else:
            my_trail = [tuple(p) for p in state.get('agent2_trail', [])]
            opp_trail = [tuple(p) for p in state.get('agent1_trail', [])]
            next_my_trail = [tuple(p) for p in next_state.get('agent2_trail', [])]
            next_opp_trail = [tuple(p) for p in next_state.get('agent1_trail', [])]

        if my_trail and next_my_trail:
            board_width, board_height = _get_board_dimensions(state)

            # Calculate space before and after
            curr_head = my_trail[-1]
            curr_space = flood_fill_area(curr_head, my_trail, opp_trail, board_width, board_height)

            next_head = next_my_trail[-1]
            next_space = flood_fill_area(next_head, next_my_trail, next_opp_trail, board_width, board_height)

            # Reward space gain
            space_delta = next_space - curr_space
            reward += space_delta * 0.01  # Small reward per unit space gained

    except Exception:
        pass  # Fallback if space calculation fails

    return reward


def perform_online_learning():
    """
    Perform conservative online learning updates using recent experiences.

    This function:
    1. Checks if we have enough experiences
    2. Samples a batch from the online replay buffer
    3. Performs gradient descent steps with gradient clipping
    4. Saves the updated model checkpoint

    Gating logic ensures stability:
    - Minimum buffer size before training
    - Limited gradient steps per call
    - Conservative learning rate
    - Gradient clipping
    """
    global ONLINE_UPDATE_COUNTER, DQN_MODEL, ONLINE_OPTIMIZER

    if not ONLINE_LEARNING_ENABLED:
        return

    if not DQN_AVAILABLE or DQN_MODEL is None or ONLINE_OPTIMIZER is None:
        return

    # Gate 1: Minimum buffer size
    if len(ONLINE_REPLAY_BUFFER) < ONLINE_MIN_BUFFER_SIZE:
        return

    # Gate 2: Confidence check - only update if we have diverse experiences
    # (This helps prevent overfitting to a single game pattern)
    if len(ONLINE_REPLAY_BUFFER) < ONLINE_MIN_BUFFER_SIZE * 1.5:
        # Early stage - be extra conservative
        num_steps = 1
    else:
        num_steps = ONLINE_GRADIENT_STEPS_PER_TURN

    try:
        gamma = 0.95  # Match training gamma

        for _ in range(num_steps):
            # Sample batch
            if len(ONLINE_REPLAY_BUFFER) < ONLINE_BATCH_SIZE:
                batch = list(ONLINE_REPLAY_BUFFER)
            else:
                import random
                batch = random.sample(list(ONLINE_REPLAY_BUFFER), ONLINE_BATCH_SIZE)

            if not batch:
                return

            # Unpack batch (same as DQNTrainer.train_step)
            states = torch.tensor([e.state for e in batch], dtype=torch.float32)
            actions = torch.tensor([e.action for e in batch], dtype=torch.long)
            rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32)
            next_states = torch.tensor([e.next_state for e in batch], dtype=torch.float32)
            dones = torch.tensor([e.done for e in batch], dtype=torch.float32)

            # Forward pass
            q_values = DQN_MODEL(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Compute target Q values (use current network as target for online learning)
            with torch.no_grad():
                next_q_values = DQN_MODEL(next_states).max(1)[0]
                target_q_values = rewards + (1 - dones) * gamma * next_q_values

            # Compute loss
            loss = nn.MSELoss()(q_values, target_q_values)

            # Backward pass with gradient clipping
            ONLINE_OPTIMIZER.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(DQN_MODEL.parameters(), 0.5)  # Conservative clipping
            ONLINE_OPTIMIZER.step()

            ONLINE_UPDATE_COUNTER += 1

        # Save checkpoint after updates to persist learning
        if ONLINE_UPDATE_COUNTER % 5 == 0:  # Save every 5 update cycles
            model_path = os.path.join(SCRIPT_DIR, 'agent_model', 'models', 'dqn_agent.pth')
            save_model(DQN_MODEL, ONLINE_OPTIMIZER, ONLINE_UPDATE_COUNTER, model_path)

    except Exception as e:
        import traceback
        print(f"⚠️  Error in online learning: {e}")
        traceback.print_exc()


def capture_experience(state, next_state, chosen_direction, action_idx, player_number, game_over=False):
    """
    Capture an experience tuple from the current game and add to online replay buffer.

    Args:
        state: Current state dict
        next_state: Next state dict after move
        chosen_direction: Direction that was chosen
        action_idx: Action index (0-3)
        player_number: 1 or 2
        game_over: Whether the game ended
    """
    global ONLINE_REPLAY_BUFFER, LAST_STATE_FEATURES, LAST_ACTION, LAST_DIRECTION

    if not ONLINE_LEARNING_ENABLED or not DQN_AVAILABLE:
        return

    try:
        # Extract features for current state and action
        state_features = extract_features(state, player_number, chosen_direction)

        # If we have a previous state, create experience
        if LAST_STATE_FEATURES is not None and LAST_ACTION is not None and LAST_DIRECTION is not None:
            # Compute reward for the transition
            reward = compute_online_reward(state, next_state, LAST_DIRECTION, player_number, game_over)

            # Create experience tuple
            experience = Experience(
                state=LAST_STATE_FEATURES,
                action=LAST_ACTION,
                reward=reward,
                next_state=state_features,
                done=game_over
            )

            ONLINE_REPLAY_BUFFER.append(experience)

        # Update last state for next transition
        if not game_over:
            LAST_STATE_FEATURES = state_features
            LAST_ACTION = action_idx
            LAST_DIRECTION = chosen_direction
        else:
            # Reset on game end
            LAST_STATE_FEATURES = None
            LAST_ACTION = None
            LAST_DIRECTION = None

    except Exception as e:
        import traceback
        print(f"⚠️  Error capturing experience: {e}")
        traceback.print_exc()


def is_two_step_path_safe(direction, state, player_number, board_width, board_height):
    """Ensure a boost won't immediately clip into a wall/trail within two steps."""
    my_trail, other_trail = _get_trails(state, player_number)
    if not my_trail:
        return False

    head = my_trail[-1]
    first = apply_direction(head, direction, board_width, board_height)
    if not is_position_safe(first, my_trail, other_trail):
        return False

    simulated_trail = list(my_trail) + [first]
    second = apply_direction(first, direction, board_width, board_height)
    return is_position_safe(second, simulated_trail, other_trail)


def should_use_boost(direction, state, player_number, boosts_remaining, turn_count, q_value,
                     board_width, board_height):
    """
    Intelligent boost strategy that adapts to game state.
    Works against any opponent strategy by balancing aggression with safety.
    """
    if boosts_remaining <= 0:
        return False

    # Get trails for analysis
    if player_number == 1:
        my_trail = state.get('agent1_trail', [])
        other_trail = state.get('agent2_trail', [])
    else:
        my_trail = state.get('agent2_trail', [])
        other_trail = state.get('agent1_trail', [])

    if not my_trail:
        return False

    my_trail = [tuple(p) if isinstance(p, list) else p for p in my_trail]
    other_trail = [tuple(p) if isinstance(p, list) else p for p in other_trail]

    # Safety first: Check 2-step path
    if not is_two_step_path_safe(direction, state, player_number, board_width, board_height):
        return False

    # Calculate strategic metrics
    from agent_model.heuristics import flood_fill_area, manhattan_distance

    head = my_trail[-1]
    new_pos = apply_direction(head, direction, board_width, board_height)
    simulated_trail = list(my_trail) + [new_pos]

    # My reachable space after boost
    my_space = flood_fill_area(new_pos, simulated_trail, other_trail, board_width, board_height)

    # Opponent's reachable space
    opponent_head = other_trail[-1] if other_trail else None
    if opponent_head:
        opponent_space = flood_fill_area(opponent_head, other_trail, simulated_trail, board_width, board_height)
        dist_to_opponent = manhattan_distance(head, opponent_head, board_width, board_height)
    else:
        opponent_space = 0
        dist_to_opponent = 999

    # Game state analysis
    max_area = board_width * board_height
    space_advantage = (my_space - opponent_space) / max_area
    board_occupancy = (len(my_trail) + len(other_trail)) / max_area

    # Minimum Q-value threshold (base safety check)
    min_q_threshold = 0.30

    # ADAPTIVE BOOST DECISION based on multiple factors:

    # Factor 1: Game phase timing
    if turn_count < 25:
        # Very early - save boosts
        return False
    elif turn_count > 180:
        # Too late - save last boost for emergency
        return boosts_remaining > 1 and q_value > 0.35

    # Factor 2: Space situation
    if my_space < 25:
        # Low space - only boost if it opens up significant area
        return q_value > 0.40 and my_space > opponent_space

    # Factor 3: Strategic opportunities
    boost_score = 0.0

    # Opportunity: Good Q-value indicates strong position
    if q_value > 0.45:
        boost_score += 0.3
    elif q_value > min_q_threshold:
        boost_score += 0.1

    # Opportunity: We have space advantage - press it
    if space_advantage > 0.1:
        boost_score += 0.25
    elif space_advantage > 0.0:
        boost_score += 0.10

    # Opportunity: We're behind - need to catch up
    if space_advantage < -0.1:
        boost_score += 0.20  # Desperate play

    # Opportunity: Board is getting crowded - grab space quickly
    if board_occupancy > 0.4:
        boost_score += 0.15

    # Opportunity: Opponent is close - create distance or cut them off
    if 3 <= dist_to_opponent <= 8:
        boost_score += 0.15

    # Opportunity: We have multiple boosts - can afford to spend
    if boosts_remaining >= 2:
        boost_score += 0.10

    # Opportunity: Good space available after boost
    if my_space > 60:
        boost_score += 0.15

    # Factor 4: Turn phase preferences
    if 40 <= turn_count <= 80:
        # Prime boost window - territory grab phase
        boost_score += 0.15
    elif 80 <= turn_count <= 140:
        # Strategic boost window - positioning phase
        boost_score += 0.10

    # Decision: Boost if score exceeds threshold AND minimum safety met
    return q_value >= min_q_threshold and boost_score >= 0.50


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server.

    The agent should update its local representation and return 200.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.

    Query params the judge sends (optional): player_number, attempt_number,
    random_moves_left, turn_count. Agents can use this to decide.

    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    where DIRECTION is UP, DOWN, LEFT, or RIGHT
    and :BOOST is optional to use a speed boost (move twice)
    """
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        state = dict(LAST_POSTED_STATE)
        prev_state = dict(LAST_POSTED_STATE)  # Store for experience capture
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        boosts_remaining = my_agent.boosts_remaining

    # -----------------HYBRID HEURISTIC + DQN LOGIC-------------------
    move, chosen_direction, action_idx = decide_move_hybrid(state, player_number, boosts_remaining)
    # -----------------end code here--------------------

    # Online learning: Capture experience and update model
    if ONLINE_LEARNING_ENABLED and DQN_AVAILABLE and chosen_direction is not None:
        # Get next state (will be available on next /send-state call, use current as proxy)
        next_state = state  # Will be updated properly on next turn
        game_over = not state.get(f'agent{player_number}_alive', True)

        # Capture experience
        capture_experience(prev_state, next_state, chosen_direction, action_idx, player_number, game_over)

        # Perform online learning updates (gated internally)
        perform_online_learning()

    return jsonify({"move": move}), 200


def decide_move_hybrid(state, player_number, boosts_remaining):
    """
    Hybrid decision logic: Heuristics filter safe moves, DQN picks the best one.

    Args:
        state: Game state dict
        player_number: 1 or 2
        boosts_remaining: Number of boosts left

    Returns:
        Move string like "UP" or "RIGHT:BOOST"
    """
    if not DQN_AVAILABLE or DQN_MODEL is None or extract_all_direction_features is None:
        return decide_move_fallback(state, player_number, boosts_remaining)

    board_width, board_height = _get_board_dimensions(state)

    try:
        safe_moves = get_safe_moves(state, player_number, board_width, board_height)

        if not safe_moves:
            return decide_move_fallback(state, player_number, boosts_remaining)

        features_by_direction = extract_all_direction_features(state, player_number, board_width, board_height)

        features_list = []
        filtered_safe_moves = []
        for direction, move_str in safe_moves:
            direction_features = features_by_direction.get(direction)
            if direction_features is None:
                continue
            features_list.append(direction_features)
            filtered_safe_moves.append((direction, move_str))

        if not filtered_safe_moves:
            return decide_move_fallback(state, player_number, boosts_remaining)

        features_tensor = torch.tensor(features_list, dtype=torch.float32)

        with torch.no_grad():
            q_values = DQN_MODEL(features_tensor)

        composite_scores = []
        direction_q_scores = []
        for idx, (direction, _) in enumerate(filtered_safe_moves):
            dir_idx = DIRECTION_TO_INDEX[direction]
            direction_q = q_values[idx, dir_idx].item()
            direction_q_scores.append(direction_q)

            features_vec = features_list[idx]
            flood_space = features_vec[5]  # Space after this move
            global_space = features_vec[6]  # Current space
            obstacle_buffer = features_vec[0]  # Distance to nearest obstacle
            opponent_distance = features_vec[7]  # Distance to opponent
            space_advantage = features_vec[15]  # My space - opponent space (normalized)
            trail_advantage = features_vec[14]  # My trail - opponent trail (normalized)

            # ANTI-TERRITORY HEURISTIC OVERRIDE
            # If we're significantly behind in space, apply strong space-maximization override
            # This counters territory-focused opponents
            if space_advantage < -0.15:  # We're losing territory badly
                # Override weights to maximize space aggressively
                territory_defense_bonus = flood_space * 2.0  # Double space weight
                direction_q_scores[idx] += territory_defense_bonus  # Boost Q-value directly

            # ADAPTIVE MULTI-STRATEGY APPROACH
            # Combines DQN learned strategy with multiple heuristic dimensions
            # This beats specialized agents by being more well-rounded

            turn_count = state.get('turn_count', 0)

            # Calculate board occupancy to determine game phase
            total_area = board_width * board_height
            occupied_ratio = (features_vec[8] + features_vec[9]) * 500 / total_area  # Denormalize trail lengths

            # Dynamic weight adjustment based on game state
            # Early game: Aggressive territory expansion
            # Mid game: Balanced strategy with tactical positioning
            # Late game: Maximize survival space while pressuring opponent

            if turn_count < 60:
                # Early game: Grab territory fast, maintain safety buffer
                space_weight = 0.60  # Increased from 0.50 for better territory control
                safety_weight = 0.20  # Reduced slightly
                advantage_weight = 0.15
                tactical_weight = 0.05
            elif turn_count < 150:
                # Mid game: Balance all factors with emphasis on space
                space_weight = 0.50  # Increased from 0.40
                safety_weight = 0.10  # Reduced
                advantage_weight = 0.25
                tactical_weight = 0.15
            else:
                # Late game: Maximize space advantage, tactical pressure
                space_weight = 0.50  # Increased from 0.45
                safety_weight = 0.08  # Reduced
                advantage_weight = 0.32  # Increased from 0.30
                tactical_weight = 0.10

            # Adjust weights based on board density
            if occupied_ratio > 0.5:
                # Board getting crowded - prioritize safety and space
                space_weight += 0.10
                safety_weight += 0.05
                tactical_weight -= 0.15

            # Calculate multi-dimensional score
            space_score = space_weight * flood_space
            safety_score = safety_weight * obstacle_buffer
            advantage_score = advantage_weight * space_advantage

            # Tactical positioning: Dynamically approach or avoid opponent
            if space_advantage > 0.05:
                # We have advantage - apply pressure by moving toward opponent
                tactical_score = tactical_weight * (1.0 - opponent_distance)
            elif space_advantage < -0.05:
                # We're behind - focus on grabbing space, avoid confrontation
                tactical_score = tactical_weight * opponent_distance
            else:
                # Even game - maintain flexible positioning
                tactical_score = tactical_weight * 0.5

            # Combine DQN output with multi-dimensional heuristics
            heuristics_bonus = space_score + safety_score + advantage_score + tactical_score

            # Add DQN weight (learned strategy)
            dqn_weight = 0.6 if turn_count < 100 else 0.5  # Trust DQN more early, balance later

            composite_scores.append(dqn_weight * direction_q + heuristics_bonus)

        best_idx = max(range(len(composite_scores)), key=lambda i: composite_scores[i])
        best_direction, best_move_str = filtered_safe_moves[best_idx]
        best_q_value = direction_q_scores[best_idx]
        best_action_idx = DIRECTION_TO_INDEX[best_direction]

        turn_count = state.get('turn_count', 0)
        use_boost = should_use_boost(
            best_direction,
            state,
            player_number,
            boosts_remaining,
            turn_count,
            best_q_value,
            board_width,
            board_height,
        )

        move_str = f"{best_move_str}:BOOST" if use_boost else best_move_str
        return move_str, best_direction, best_action_idx

    except Exception as e:
        import traceback

        print(f"⚠️  Error in DQN inference: {e}")
        traceback.print_exc()
        return decide_move_fallback(state, player_number, boosts_remaining)


def decide_move_fallback(state, player_number, boosts_remaining):
    """
    Fallback rule-based decision logic when DQN is not available.
    Uses heuristics only.

    Args:
        state: Game state dict
        player_number: 1 or 2
        boosts_remaining: Number of boosts left

    Returns:
        Tuple of (move_string, direction, action_idx)
    """
    my_trail, other_trail = _get_trails(state, player_number)
    if not my_trail:
        return "RIGHT", Direction.RIGHT, DIRECTION_TO_INDEX[Direction.RIGHT]

    board_width, board_height = _get_board_dimensions(state)
    safe_moves = get_safe_moves(state, player_number, board_width, board_height)

    if not safe_moves:
        return "RIGHT", Direction.RIGHT, DIRECTION_TO_INDEX[Direction.RIGHT]

    current_dir = get_current_direction(my_trail)
    chosen_direction = None
    chosen_move_str = None

    for direction, move_str in safe_moves:
        if direction == current_dir:
            chosen_direction = direction
            chosen_move_str = move_str
            break

    if chosen_direction is None:
        head = my_trail[-1]
        best_area = -1
        best_move = safe_moves[0]

        for direction, move_str in safe_moves:
            new_pos = apply_direction(head, direction, board_width, board_height)
            simulated_trail = list(my_trail) + [new_pos]
            area = flood_fill_area(new_pos, simulated_trail, other_trail, board_width, board_height)

            if area > best_area:
                best_area = area
                best_move = (direction, move_str)

        chosen_direction, chosen_move_str = best_move
        area_confidence = min(1.0, max(0.0, best_area / (board_width * board_height)))
    else:
        area_confidence = 0.5

    turn_count = state.get('turn_count', 0)
    q_estimate = 0.4 + 0.6 * area_confidence
    use_boost = should_use_boost(
        chosen_direction,
        state,
        player_number,
        boosts_remaining,
        turn_count,
        q_estimate,
        board_width,
        board_height,
    )

    move_str = f"{chosen_move_str}:BOOST" if use_boost else chosen_move_str
    action_idx = DIRECTION_TO_INDEX[chosen_direction]
    return move_str, chosen_direction, action_idx


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state.

    We update local state for record-keeping and return OK.
    """
    global LAST_STATE_FEATURES, LAST_ACTION, LAST_DIRECTION, GAME_STATS

    data = request.get_json()
    if data:
        _update_local_game_from_post(data)

        # Capture final experience if online learning enabled
        if ONLINE_LEARNING_ENABLED and DQN_AVAILABLE:
            player_number = 1  # Default, could be passed in data
            if LAST_STATE_FEATURES is not None and LAST_ACTION is not None:
                # Final state with game_over=True
                final_state = dict(LAST_POSTED_STATE)
                capture_experience(final_state, final_state, LAST_DIRECTION, LAST_ACTION, player_number, game_over=True)

            # Perform final batch of online learning
            perform_online_learning()

            # Save checkpoint at end of game
            if DQN_MODEL is not None and ONLINE_OPTIMIZER is not None:
                model_path = os.path.join(SCRIPT_DIR, 'agent_model', 'models', 'dqn_agent.pth')
                save_model(DQN_MODEL, ONLINE_OPTIMIZER, ONLINE_UPDATE_COUNTER, model_path)
                print(f"📊 Game ended - checkpoint saved ({len(ONLINE_REPLAY_BUFFER)} experiences in buffer)")

            # Reset per-game state
            LAST_STATE_FEATURES = None
            LAST_ACTION = None
            LAST_DIRECTION = None

            # Update game statistics
            GAME_STATS['games_played'] += 1

    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    app.run(host="0.0.0.0", port=port, debug=True)
