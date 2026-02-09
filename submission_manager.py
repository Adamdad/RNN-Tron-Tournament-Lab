"""
Submission Manager - Load Student Agents
"""

import os
import sys
import torch
import importlib.util
from pathlib import Path


class SubmissionManager:
    """Manage student submissions"""
    
    def __init__(self, submissions_dir="submissions"):
        self.submissions_dir = Path(submissions_dir)
        self.submissions_dir.mkdir(exist_ok=True)
    
    def load_all_agents(self):
        """Load all Agents"""
        agents = {}
        
        for py_file in self.submissions_dir.glob("*_agent.py"):
            name = py_file.stem
            agent = self._load_agent(py_file)
            if agent:
                agents[name] = agent
                print(f"✓ Loaded {name}")
        
        return agents
    
    def _load_agent(self, py_file):
        """Load single Agent"""
        try:
            # Dynamic import
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[py_file.stem] = module
            spec.loader.exec_module(module)
            
            # Find StudentAgent class
            if not hasattr(module, 'StudentAgent'):
                return None
            
            agent = module.StudentAgent()
            
            # Load weights
            pth_file = py_file.with_suffix('.pth')
            if pth_file.exists():
                state_dict = torch.load(pth_file, map_location='cpu')
                agent.load_state_dict(state_dict)
                agent.eval()
            
            return agent
            
        except Exception as e:
            print(f"✗ Failed to load {py_file.name}: {e}")
            return None


if __name__ == "__main__":
    manager = SubmissionManager()
    agents = manager.load_all_agents()
    print(f"\nLoaded {len(agents)} agents")
