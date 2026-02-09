"""
锦标赛运行器
"""

import numpy as np
import torch
from tron_env import BlindTronEnv
from base_agent import BaseAgent, RandomAgent, ExampleAgent


class Tournament:
    def __init__(self, grid_size=20):
        self.grid_size = grid_size
    
    def run_game(self, agent1, agent2):
        """运行单局游戏"""
        env = BlindTronEnv(grid_size=self.grid_size, render_mode=False)
        obs1, obs2 = env.reset()
        agent1.reset()
        agent2.reset()
        
        done = False
        steps = 0
        max_steps = self.grid_size * self.grid_size * 2
        
        while not done and steps < max_steps:
            with torch.no_grad():
                a1 = agent1.get_action(obs1)
                a2 = agent2.get_action(obs2)
            obs1, obs2, done, winner = env.step(a1, a2)
            steps += 1
        
        return winner
    
    def run_match(self, agent1, agent2, name1, name2, num_games=3):
        """运行多局比赛"""
        print(f"\n{name1} vs {name2}")
        
        wins = {name1: 0, name2: 0, "draw": 0}
        for i in range(num_games):
            winner = self.run_game(agent1, agent2)
            if winner == 1:
                wins[name1] += 1
            elif winner == 2:
                wins[name2] += 1
            else:
                wins["draw"] += 1
        
        print(f"  Result: {wins}")
        return wins
    
    def run_tournament(self, agents, games_per_pair=3):
        """运行循环赛"""
        names = list(agents.keys())
        scores = {name: {"points": 0, "wins": 0} for name in names}
        
        print(f"\n{'='*60}")
        print(f"Tournament: {len(names)} agents, {games_per_pair} games each")
        print(f"{'='*60}")
        
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                name1, name2 = names[i], names[j]
                wins = self.run_match(agents[name1], agents[name2], name1, name2, games_per_pair)
                
                # 计分
                if wins[name1] > wins[name2]:
                    scores[name1]["points"] += 3
                    scores[name1]["wins"] += 1
                elif wins[name2] > wins[name1]:
                    scores[name2]["points"] += 3
                    scores[name2]["wins"] += 1
                else:
                    scores[name1]["points"] += 1
                    scores[name2]["points"] += 1
        
        # 排序
        rankings = sorted(scores.items(), key=lambda x: (-x[1]["points"], -x[1]["wins"]))
        
        print(f"\n{'='*60}")
        print("Final Rankings:")
        print(f"{'='*60}")
        for rank, (name, score) in enumerate(rankings, 1):
            print(f"{rank}. {name}: {score['points']} points, {score['wins']} wins")
        
        return rankings


if __name__ == "__main__":
    # 测试
    agents = {
        "Example": ExampleAgent(),
        "Random": RandomAgent(),
    }
    
    tourney = Tournament()
    tourney.run_tournament(agents, games_per_pair=3)
