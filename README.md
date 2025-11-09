# Dong Sun Run - Case Closed Agent Submission

This repository contains our agent submission for the Case Closed competition.

## Agent Architecture

- **Hybrid DQN + Heuristic Approach**: Combines deep reinforcement learning with rule-based heuristics
- **Multi-Strategy**: Adapts strategy based on game phase (early/mid/late game)

## Files

- `agent.py` - Main agent logic and Flask server
- `requirements.txt` - Python dependencies
- `agent_model/` - Helper modules for DQN and feature extraction
  - `dqn_model.py` - Neural network model
  - `features.py` - Feature extraction from game state
  - `heuristics.py` - Safety checks and strategic heuristics
  - `models/dqn_agent.pth` - Trained model weights

## Running Locally

```bash
pip install -r requirements.txt
python agent.py
```

The agent will start a Flask server on port 8080.
