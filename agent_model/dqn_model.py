"""
Lightweight DQN model for Case Closed agent decision making.
Uses a simple 3-layer dense network optimized for CPU inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CaseClosedDQN(nn.Module):
    """
    Deep Q-Network for Case Closed game.

    Input: Feature vector (configurable size, default 16)
    Output: Q-values for 4 actions (UP, DOWN, LEFT, RIGHT)
    """

    def __init__(self, input_size=16, hidden_size=64):
        super(CaseClosedDQN, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 4)  # 4 actions: UP, DOWN, LEFT, RIGHT

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Tensor of shape (batch_size, input_size)

        Returns:
            Q-values of shape (batch_size, 4)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation on output (raw Q-values)
        return x


def load_model(model_path='models/dqn_agent.pth', input_size=16, hidden_size=64, device='cpu'):
    """
    Load a trained DQN model from disk.

    Args:
        model_path: Path to the saved model checkpoint
        input_size: Size of input feature vector
        hidden_size: Size of hidden layers
        device: Device to load model on (forced to 'cpu' for competition compliance)

    Returns:
        Loaded model in eval mode
    """
    # Force CPU device for competition compliance
    if device != 'cpu':
        print(f"⚠️  Warning: device '{device}' requested but forcing 'cpu' for competition compliance")
        device = 'cpu'

    model = CaseClosedDQN(input_size=input_size, hidden_size=hidden_size)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print(f"✅ Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"⚠️  Model file not found at {model_path}. Using untrained model.")
    except Exception as e:
        print(f"⚠️  Error loading model: {e}. Using untrained model.")

    return model


def save_model(model, optimizer, episode, model_path='models/dqn_agent.pth'):
    """
    Save model checkpoint to disk.

    Args:
        model: The DQN model to save
        optimizer: The optimizer state to save
        episode: Current training episode number
        model_path: Path to save the checkpoint
    """
    import os
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode': episode,
    }, model_path)
    print(f"💾 Model saved to {model_path} at episode {episode}")
