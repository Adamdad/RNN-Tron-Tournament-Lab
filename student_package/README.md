# RNN Tron Challenge - Student Package

Welcome to the RNN Tron Challenge! Train an RNN Agent to compete in the Tron game.

## What's Included

```
student_package/
├── README.md              # This file
├── student_template.py    # Main template - EDIT THIS FILE!
├── student_template.ipynb # Jupyter notebook for visualization & testing
├── tron_env.py           # Game environment (for local testing)
├── requirements.txt      # Python dependencies
├── packages.txt          # System dependencies (for pygame)
├── train_X.npy          # Training data (provided by instructor)
├── train_Y.npy          # Training labels (provided by instructor)
└── submissions/          # Where to save your trained model
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

For Jupyter notebook:
```bash
pip install jupyter ipython
```

On Linux, you may also need:
```bash
sudo apt-get install libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev
```

### 2. Edit the Template

**Edit `student_template.py`** and fill in:

```python
# Lines 10-15: Your information
STUDENT_INFO = {
    "name": "Your Name",
    "student_id": "Your Student ID",
    "team_name": "Your Team Name",
    "description": "Brief description of your approach"
}

# Lines 19-28: Your model architecture
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Define your network
        self.lstm = nn.LSTM(10, 64, batch_first=True)
        self.fc = nn.Linear(64, 4)
```

### 3. Development Workflow

#### Option A: Jupyter Notebook (Recommended for Development)

The notebook **imports from your template**, so you don't need to duplicate code:

```python
from student_template import STUDENT_INFO, MyModel, StudentAgent
```

**Benefits:**
- Interactive training with parameter adjustment
- Real-time game visualization
- Step-by-step debugging

**Usage:**
```bash
jupyter notebook student_template.ipynb
```

**Notebook Steps:**
1. Run all cells to import your agent
2. Adjust training parameters (epochs, lr, batch_size)
3. Execute training cell
4. Watch your agent play in the visualization

#### Option B: Python Script (For Final Training)

```bash
python student_template.py --train --epochs 20
```

## Development Tips

1. **Start with the notebook** to visualize and understand the game
2. **Experiment in the notebook** with different architectures
3. **Use the script** for final training with more epochs
4. **Keep `student_template.py` as your source of truth** - the notebook imports from it

## Understanding the Code

### student_template.py (EDIT THIS)
Contains three key components you must define:

```python
STUDENT_INFO = {...}      # Your team info
class MyModel(nn.Module):  # Your neural network
class StudentAgent:        # The agent wrapper (DON'T RENAME)
```

### student_template.ipynb (VISUALIZATION ONLY)
- Imports your code from `student_template.py`
- Provides training and visualization cells
- Helps you debug and understand your agent

**Key imports:**
```python
from student_template import STUDENT_INFO, MyModel, StudentAgent
```

## Submission

Submit **two files**:
1. `student_template.py` (your code with STUDENT_INFO, MyModel, StudentAgent)
2. `submissions/your_name_agent.pth` (your trained weights)

## Constraints

- Maximum 100,000 parameters
- Must use provided training data
- Class name must be `StudentAgent`
- No hard-coded strategies

## Observation Format (10-dim vector)

- **0-7**: Distance to obstacle in 8 directions (N, NE, E, SE, S, SW, W, NW), normalized 0-1
- **8-9**: Normalized player coordinates (x, y)

## Action Space

- **0**: UP
- **1**: DOWN
- **2**: LEFT
- **3**: RIGHT

## Getting Help

1. Check comments in `student_template.py`
2. Use the notebook visualizations to debug
3. Ask questions during office hours

Good luck!
