"""
Base Agent Interface
Students must implement the StudentAgent class
"""

import torch
import torch.nn as nn
import numpy as np


class BaseAgent(nn.Module):
    """Base Agent class"""
    
    def __init__(self, name="Agent"):
        super().__init__()
        self.name = name
        self.hidden = None
        self.model = self.build_model()
    
    def build_model(self):
        """Build model - subclasses must implement"""
        raise NotImplementedError
    
    def reset(self):
        """Reset state"""
        self.hidden = None
    
    def get_action(self, obs):
        """Get action"""
        x = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            if self.hidden is not None:
                out, self.hidden = self.model(x, self.hidden)
            else:
                out = self.model(x)
                if isinstance(out, tuple):
                    out, self.hidden = out
            return torch.argmax(out, dim=-1).item()


class ExampleModel(nn.Module):
    """Example model"""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(10, 64, batch_first=True)
        self.fc = nn.Linear(64, 4)
    
    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        return self.fc(out[:, -1, :]), hidden


class ExampleAgent(BaseAgent):
    """Example Agent"""
    def __init__(self):
        super().__init__("Example")
    
    def build_model(self):
        return ExampleModel()
    
    def forward(self, x, hidden=None):
        return self.model(x, hidden)


class RandomAgent(BaseAgent):
    """Random Agent"""
    def __init__(self):
        super().__init__("Random")
        self.model = None
    
    def build_model(self):
        return None
    
    def get_action(self, obs):
        import random
        return random.randint(0, 3)
    
    def forward(self, x, hidden=None):
        return None, None
