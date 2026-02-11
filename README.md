# RNN Tron Tournament Lab

A tournament system for students to train RNN Agents to battle in the Tron game.

## File Structure

```
lab/
├── base_agent.py           # Base Agent class (68 lines)
├── student_template.py     # Student template (95 lines, includes training code)
├── student_template.ipynb  # Jupyter notebook version with visualizations
├── generate_data.py        # Generate expert data (93 lines)
├── tournament_runner.py    # Tournament runner (92 lines)
├── tournament_app.py       # Streamlit interface (116 lines)
├── submission_manager.py   # Submission manager (54 lines)
├── tron_env.py            # Game environment (209 lines)
└── submissions/           # Student submissions directory
```

## Quick Start

### 1. Teacher Generates Data

```bash
python generate_data.py --games 1000
```

### 2. Student Development

Students can use either the Python script or Jupyter notebook:

**Option A: Python Script**
```bash
cp student_template.py submissions/your_name_agent.py
python submissions/your_name_agent.py --train --epochs 20
```

**Option B: Jupyter Notebook (Recommended)**
```bash
jupyter notebook student_template.ipynb
```

The notebook includes:
- Training code
- Game simulation with video visualization
- Final game state display

### 3. Run Tournament

```bash
# Command line
python tournament_runner.py

# Or Streamlit interface
streamlit run tournament_app.py
```

## Core Class Documentation

### StudentAgent (Student must implement)

```python
class StudentAgent(nn.Module):
    def __init__(self):
        self.name = "TeamName"
        self.model = MyModel()  # Your RNN model
        self.hidden = None
    
    def reset(self):
        """Called before each game starts"""
        self.hidden = None
    
    def get_action(self, obs):
        """Select action based on observation (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)"""
        # obs: numpy array (10,)
        # Return: int action
```

## Dependencies

```bash
pip install torch numpy pygame streamlit pillow matplotlib
```

For Jupyter notebook:
```bash
pip install jupyter ipython
```

## Observation Format

The input observation is a 10-dimensional vector:
- **0-7**: Distance to nearest obstacle in 8 directions (N, NE, E, SE, S, SW, W, NW), normalized 0-1
- **8-9**: Normalized player coordinates (x, y)

## Action Space

- **0**: UP
- **1**: DOWN
- **2**: LEFT
- **3**: RIGHT
