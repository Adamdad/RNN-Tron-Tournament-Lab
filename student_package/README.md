# RNN Tron Challenge - Student Package

Welcome to the RNN Tron Challenge! Train an RNN Agent to compete in the Tron game.

## What's Included

```
student_package/
├── README.md              # This file
├── student_template.py    # Python template - MODIFY THIS!
├── student_template.ipynb # Jupyter notebook with visualizations
├── tron_env.py           # Game environment (for local testing)
├── requirements.txt      # Python dependencies
├── packages.txt          # System dependencies (for pygame)
├── train_X.npy          # Training data (provided by instructor)
├── train_Y.npy          # Training labels (provided by instructor)
└── submissions/          # Where to save your trained model
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

For Jupyter notebook:
```bash
pip install jupyter ipython
```

On Linux, you may also need system dependencies for pygame:
```bash
# Ubuntu/Debian
sudo apt-get install libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev
```

## Development Options

### Option A: Jupyter Notebook (Recommended)

The notebook provides an interactive environment with built-in visualizations:

```bash
jupyter notebook student_template.ipynb
```

**Notebook Features:**
- Interactive training with real-time feedback
- Game simulation video - watch your agent play
- Final game state visualization
- Step-by-step code cells for easy debugging

### Option B: Python Script

For command-line development:

```bash
python student_template.py --train --epochs 20
```

## Steps to Complete

### 1. Fill in Your Information

**In the template (lines 10-15):**
```python
STUDENT_INFO = {
    "name": "Your Name",
    "student_id": "Your Student ID",
    "team_name": "Your Team Name",
    "description": "Brief description of your approach"
}
```

### 2. Design Your Model

**Modify `MyModel` class (lines 19-28):**
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Define your network architecture
        # Default is a simple LSTM, but you can modify it!
        self.lstm = nn.LSTM(10, 64, batch_first=True)
        self.fc = nn.Linear(64, 4)
```

### 3. Train Your Model

**Jupyter Notebook:**
- Run all cells up to "Main Program - Train Your Model"
- Adjust epochs, learning rate, and batch size
- Execute the training cell

**Python Script:**
```bash
python student_template.py --train --epochs 20
```

This saves your model to `submissions/your_name_agent.pth`

### 4. Visualize Your Agent (Notebook Only)

The notebook includes visualization cells that:
- Show a video of your agent playing
- Display the final game state
- Help you understand your agent's behavior

### 5. Test Your Agent Locally

```python
from student_template import StudentAgent
from tron_env import BlindTronEnv
import random

agent = StudentAgent()
env = BlindTronEnv(render_mode=True)

# Run a test game
obs1, obs2 = env.reset()
agent.reset()
done = False

while not done:
    action = agent.get_action(obs1)
    opponent_action = random.randint(0, 3)
    obs1, obs2, done, winner = env.step(action, opponent_action)

print(f"Game over! Winner: {winner}")
```

### 6. Submit Your Solution

Submit **two files**:
1. `student_template.py` (or your renamed version)
2. `submissions/your_name_agent.pth` (trained weights)

## Tips for Success

1. **Understand the observation** (10-dim vector):
   - 8 direction distances (N, NE, E, SE, S, SW, W, NW) - normalized 0-1
   - 2 normalized coordinates (x, y)

2. **Action space** (4 actions):
   - 0 = UP
   - 1 = DOWN
   - 2 = LEFT
   - 3 = RIGHT

3. **Model constraints**:
   - Maximum 100,000 parameters
   - Must implement `StudentAgent` class
   - Must have `get_action()` and `reset()` methods

4. **Try different architectures**:
   - LSTM/GRU with different hidden sizes
   - Multiple LSTM layers
   - Add dropout for regularization
   - Try attention mechanisms

5. **Training tips**:
   - Start with default settings
   - Monitor accuracy on training data
   - Try different learning rates (0.001, 0.0005, 0.0001)
   - Increase epochs gradually (watch for overfitting)

## File Descriptions

### student_template.py / student_template.ipynb
Your main files to modify. Contains:
- `STUDENT_INFO`: Your team information
- `MyModel`: Your neural network architecture
- `StudentAgent`: The agent class (do NOT rename!)
- `train()`: Training function
- Visualizations (notebook only)

### tron_env.py
The Tron game environment:
- 20x20 grid
- Two players (you vs opponent)
- Ray-casting observations (8 directions)
- Collision detection
- Rendering support

### train_X.npy & train_Y.npy
Training data for imitation learning (expert demonstrations).

## Rules & Constraints

1. You can modify `student_template.py` or `student_template.ipynb`
2. Class name must remain `StudentAgent`
3. Model must have fewer than 100,000 parameters
4. Training must use the provided dataset
5. No hard-coding of specific strategies

## Evaluation

Your agent competes in a round-robin tournament:
- **Win**: 3 points
- **Draw**: 1 point
- **Loss**: 0 points

Final ranking based on total points and number of wins.

## Getting Help

1. Check the comments in the template files
2. Use the notebook visualizations to debug
3. Ask questions during office hours
4. Discuss strategies with classmates (but don't share code!)

Good luck and have fun!
