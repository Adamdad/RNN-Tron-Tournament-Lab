"""
Base Agent Interface
学生必须实现 StudentAgent 类
"""

import torch
import torch.nn as nn
import numpy as np


class BaseAgent(nn.Module):
    """Agent基类"""
    
    def __init__(self, name="Agent"):
        super().__init__()
        self.name = name
        self.hidden = None
        self.model = self.build_model()
    
    def build_model(self):
        """构建模型 - 子类必须实现"""
        raise NotImplementedError
    
    def reset(self):
        """重置状态"""
        self.hidden = None
    
    def get_action(self, obs):
        """获取动作"""
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
    """示例模型"""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(10, 64, batch_first=True)
        self.fc = nn.Linear(64, 4)
    
    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        return self.fc(out[:, -1, :]), hidden


class ExampleAgent(BaseAgent):
    """示例Agent"""
    def __init__(self):
        super().__init__("Example")
    
    def build_model(self):
        return ExampleModel()
    
    def forward(self, x, hidden=None):
        return self.model(x, hidden)


class RandomAgent(BaseAgent):
    """随机Agent"""
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
