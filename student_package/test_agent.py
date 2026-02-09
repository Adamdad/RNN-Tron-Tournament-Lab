"""
Test your agent locally before submission
Usage: python test_agent.py
"""

from student_template import StudentAgent
from tron_env import BlindTronEnv
import random
import time

def test_agent(num_games=5):
    """Test your agent against random opponents"""
    agent = StudentAgent()
    env = BlindTronEnv(render_mode=True)
    
    wins = 0
    draws = 0
    losses = 0
    
    print(f"Testing agent: {agent.name}")
    print(f"Model parameters: {sum(p.numel() for p in agent.parameters())}")
    print(f"Running {num_games} games...\n")
    
    for game in range(num_games):
        obs1, obs2 = env.reset()
        agent.reset()
        done = False
        steps = 0
        
        while not done and steps < 400:
            # Your agent's action
            action1 = agent.get_action(obs1)
            # Random opponent
            action2 = random.randint(0, 3)
            
            obs1, obs2, done, winner = env.step(action1, action2)
            steps += 1
            time.sleep(0.05)  # Slow down for visibility
        
        if winner == 1:
            wins += 1
            result = "WIN"
        elif winner == 2:
            losses += 1
            result = "LOSS"
        else:
            draws += 1
            result = "DRAW"
        
        print(f"Game {game+1}: {result} ({steps} steps)")
    
    print(f"\n{'='*40}")
    print(f"Results: {wins} wins, {draws} draws, {losses} losses")
    print(f"Win rate: {100*wins/num_games:.1f}%")

if __name__ == "__main__":
    test_agent(num_games=5)
