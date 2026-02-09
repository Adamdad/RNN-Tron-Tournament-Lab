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
    
    def get_all_submissions(self):
        """Get list of all submission names"""
        submissions = []
        for py_file in self.submissions_dir.glob("*_agent.py"):
            name = py_file.stem.replace("_agent", "")
            submissions.append(name)
        return sorted(submissions)
    
    def delete_submission(self, name):
        """Delete a submission by name"""
        try:
            # Find files
            py_file = self.submissions_dir / f"{name}_agent.py"
            pth_file = self.submissions_dir / f"{name}_agent.pth"
            
            deleted = False
            
            # Delete .py file
            if py_file.exists():
                py_file.unlink()
                deleted = True
                print(f"✓ Deleted {py_file.name}")
            
            # Delete .pth file
            if pth_file.exists():
                pth_file.unlink()
                deleted = True
                print(f"✓ Deleted {pth_file.name}")
            
            # Also delete backup files if exist
            for backup_file in self.submissions_dir.glob(f"{name}_agent_*.py"):
                backup_file.unlink()
                print(f"✓ Deleted backup {backup_file.name}")
            
            for backup_file in self.submissions_dir.glob(f"{name}_agent_*.pth"):
                backup_file.unlink()
                print(f"✓ Deleted backup {backup_file.name}")
            
            return deleted
            
        except Exception as e:
            print(f"✗ Failed to delete {name}: {e}")
            return False


if __name__ == "__main__":
    manager = SubmissionManager()
    agents = manager.load_all_agents()
    print(f"\nLoaded {len(agents)} agents")
