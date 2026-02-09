# RNN Tron Tournament Lab

A tournament system for students to train RNN Agents to battle in the Tron game.

## File Structure

```
lab/
├── base_agent.py           # Base Agent class (68 lines)
├── student_template.py     # Student template (95 lines, includes training code)
├── generate_data.py        # Generate expert data (93 lines)
├── tournament_runner.py    # Tournament runner (92 lines)
├── tournament_app.py       # Streamlit interface (116 lines)
├── submission_manager.py   # Submission manager (54 lines)
├── tron_env.py            # Game environment (retained)
└── submissions/           # Student submissions directory
```

## Usage Workflow

### 1. Teacher Generates Data

```bash
python generate_data.py --games 1000
```

### 2. Student Development

Copy the template and modify:

```bash
cp student_template.py submissions/your_name_agent.py
```

Modify `STUDENT_INFO` and `MyModel` class, then train:

```bash
python submissions/your_name_agent.py --train --epochs 20
```

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

## Simplification Notes

- **base_agent.py**: From 202 lines -> 68 lines
- **student_template.py**: From 470 lines -> 95 lines  
- **generate_data.py**: From 504 lines -> 93 lines
- **tournament_runner.py**: From 312 lines -> 92 lines
- **tournament_app.py**: From 315 lines -> 116 lines
- **submission_manager.py**: From 330 lines -> 54 lines

Total reduced from 2133 lines to 518 lines (75% reduction)

## Dependencies

```bash
pip install torch numpy pygame streamlit pillow
```
