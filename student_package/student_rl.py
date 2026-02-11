"""
Hybrid RL Agent: Imitation Learning + Reinforcement Learning
1. Pre-train on expert data (supervised)
2. Fine-tune with RL (self-play)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


# ========== Configuration ==========
CONFIG = {
    'episodes': 2000,
    'il_epochs': 20,        # Imitation learning epochs
    'rl_episodes': 1000,     # RL episodes after IL
    'lr': 3e-4,
    'gamma': 0.99,
    'max_steps': 300,
    'batch_size': 64
}

STUDENT_INFO = {
    "name": "Xingyi Yang",
    "student_id": "ABC",
    "team_name": "DSAI5027_Hybrid",
    "description": "IL + RL"
}


# ========== Network ==========
class PolicyNetwork(nn.Module):
    """Simple policy network with LSTM"""
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.policy_head = nn.Linear(hidden_dim, output_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        encoded = self.encoder(x)
        lstm_out, hidden = self.lstm(encoded, hidden)
        features = lstm_out[:, -1, :]
        
        return self.policy_head(features), self.value_head(features).squeeze(-1), hidden


# ========== Agent ==========
class StudentAgent(nn.Module):
    """Student agent for submission"""
    def __init__(self):
        super().__init__()
        self.name = STUDENT_INFO.get("team_name") or STUDENT_INFO["name"]
        self.info = STUDENT_INFO
        self.model = PolicyNetwork()
        self.hidden = None
    
    def reset(self):
        self.hidden = None
    
    def get_action(self, obs):
        """Get action for observation (deterministic)"""
        x = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            logits, _, self.hidden = self.model(x, self.hidden)
        return logits.argmax(dim=-1).item()
    
    def forward(self, x, hidden=None):
        return self.model(x, hidden)


# ========== Imitation Learning ==========
def train_imitation(agent, data_path="train_X.npy", labels_path="train_Y.npy",
                   epochs=20, lr=3e-4, batch_size=64):
    """Pre-train on expert demonstrations"""
    print("=" * 50)
    print("Phase 1: Imitation Learning")
    print("=" * 50)
    
    # Load data
    X = np.load(data_path)
    Y = np.load(labels_path)
    print(f"Loaded {len(X)} expert samples")
    
    # Prepare data
    X_t = torch.FloatTensor(X)
    Y_t = torch.LongTensor(Y)
    dataset = torch.utils.data.TensorDataset(X_t, Y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            
            # Forward
            logits, _, _ = agent.model(batch_x)
            loss = criterion(logits, batch_y)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == batch_y).sum().item()
        
        acc = correct / len(X) * 100
        avg_loss = total_loss / len(loader)
        
        if acc > best_acc:
            best_acc = acc
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")
    
    print(f"\nImitation learning complete. Best Acc: {best_acc:.2f}%")
    print(f"Parameters: {sum(p.numel() for p in agent.parameters())}")
    print()


# ========== Reinforcement Learning ==========
def collect_episode(env, agent, optimizer):
    """Collect one episode and train with REINFORCE"""
    obs1, obs2 = env.reset()
    
    log_probs_p1, values_p1, rewards_p1 = [], [], []
    log_probs_p2, values_p2, rewards_p2 = [], [], []
    
    hidden_p1, hidden_p2 = None, None
    done = False
    step = 0
    
    while not done and step < CONFIG['max_steps']:
        obs1_t = torch.from_numpy(obs1).float().unsqueeze(0)
        obs2_t = torch.from_numpy(obs2).float().unsqueeze(0)
        
        # Player 1
        logits_p1, value_p1, hidden_p1 = agent.model(obs1_t, hidden_p1)
        probs_p1 = F.softmax(logits_p1, dim=-1)
        dist_p1 = torch.distributions.Categorical(probs_p1)
        action_p1 = dist_p1.sample()
        log_prob_p1 = dist_p1.log_prob(action_p1)
        
        # Player 2
        logits_p2, value_p2, hidden_p2 = agent.model(obs2_t, hidden_p2)
        probs_p2 = F.softmax(logits_p2, dim=-1)
        dist_p2 = torch.distributions.Categorical(probs_p2)
        action_p2 = dist_p2.sample()
        log_prob_p2 = dist_p2.log_prob(action_p2)
        
        # Step environment
        obs1_next, obs2_next, done, winner = env.step(action_p1.item(), action_p2.item())
        
        # Rewards
        if done:
            if winner == 1:
                reward_p1, reward_p2 = 1.0, -1.0
            elif winner == 2:
                reward_p1, reward_p2 = -1.0, 1.0
            else:
                reward_p1 = reward_p2 = 0.0
        else:
            reward_p1 = reward_p2 = 0.01
        
        # Store
        log_probs_p1.append(log_prob_p1)
        values_p1.append(value_p1)
        rewards_p1.append(reward_p1)
        
        log_probs_p2.append(log_prob_p2)
        values_p2.append(value_p2)
        rewards_p2.append(reward_p2)
        
        obs1, obs2 = obs1_next, obs2_next
        step += 1
    
    # Compute returns
    returns_p1 = []
    returns_p2 = []
    R = 0
    for r in reversed(rewards_p1):
        R = r + CONFIG['gamma'] * R
        returns_p1.insert(0, R)
    R = 0
    for r in reversed(rewards_p2):
        R = r + CONFIG['gamma'] * R
        returns_p2.insert(0, R)
    
    returns_p1 = torch.tensor(returns_p1)
    returns_p2 = torch.tensor(returns_p2)
    
    # Normalize
    returns_p1 = (returns_p1 - returns_p1.mean()) / (returns_p1.std() + 1e-8)
    returns_p2 = (returns_p2 - returns_p2.mean()) / (returns_p2.std() + 1e-8)
    
    # Loss
    values_t_p1 = torch.cat(values_p1)
    log_probs_t_p1 = torch.cat(log_probs_p1)
    advantages_p1 = returns_p1 - values_t_p1.detach()
    loss_p1 = -(log_probs_t_p1 * advantages_p1).mean() + 0.5 * F.mse_loss(values_t_p1, returns_p1)
    
    values_t_p2 = torch.cat(values_p2)
    log_probs_t_p2 = torch.cat(log_probs_p2)
    advantages_p2 = returns_p2 - values_t_p2.detach()
    loss_p2 = -(log_probs_t_p2 * advantages_p2).mean() + 0.5 * F.mse_loss(values_t_p2, returns_p2)
    
    loss = loss_p1 + loss_p2
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return sum(rewards_p1), sum(rewards_p2), winner, step, loss.item()


def train_rl(env_class, agent, episodes=1000):
    """Fine-tune with RL"""
    print("=" * 50)
    print("Phase 2: Reinforcement Learning")
    print("=" * 50)
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=CONFIG['lr'])
    
    wins = [0, 0, 0]
    all_rewards = []
    
    for episode in range(episodes):
        env = env_class(render_mode=False)
        r1, r2, winner, steps, loss = collect_episode(env, agent, optimizer)
        env.close()
        
        wins[winner] += 1
        all_rewards.append((r1 + r2) / 2)
        
        if (episode + 1) % 200 == 0:
            avg_reward = np.mean(all_rewards[-200:])
            total = sum(wins)
            win_rate = wins[1] / total * 100 if total > 0 else 0
            
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Reward: {avg_reward:.3f} | Steps: {steps}")
            print(f"  Win% P1: {win_rate:.1f}% (W{wins[1]}/L{wins[2]}/D{wins[0]})")
            print(f"  Loss: {loss:.3f}")
            print()
            wins = [0, 0, 0]


# ========== Training Pipeline ==========
def train(env_class, il_epochs=20, rl_episodes=1000):
    """Full training: IL -> RL"""
    agent = StudentAgent()
    
    # Phase 1: Imitation Learning
    train_imitation(agent, epochs=il_epochs)
    
    # Phase 2: Reinforcement Learning
    train_rl(env_class, agent, episodes=rl_episodes)
    
    # Save
    os.makedirs("submissions", exist_ok=True)
    path = f"submissions/{STUDENT_INFO['name'].replace(' ', '_').lower()}_agent_hybrid.pth"
    torch.save(agent.state_dict(), path)
    print(f"\nSaved to {path}")


def play(env_class):
    """Play trained agent vs random"""
    import sys
    sys.path.insert(0, '.')
    from base_agent import RandomAgent
    
    agent = StudentAgent()
    save_path = f"submissions/{STUDENT_INFO['name'].replace(' ', '_').lower()}_agent_hybrid.pth"
    
    if os.path.exists(save_path):
        agent.load_state_dict(torch.load(save_path))
        print("Loaded trained model")
    else:
        print("Warning: No trained model found")
    
    random_agent = RandomAgent()
    env = env_class(render_mode=True)
    
    obs1, obs2 = env.reset()
    agent.reset()
    
    done = False
    step = 0
    
    print("\nPlaying... Close window to exit")
    import pygame
    
    while not done and step < CONFIG['max_steps']:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        if done:
            break
        
        action1 = agent.get_action(obs1)
        action2 = random_agent.get_action(obs2)
        
        obs1, obs2, done, winner = env.step(action1, action2)
        step += 1
    
    result = {1: "RL Agent wins!", 2: "Random wins!", 0: "Draw!"}.get(winner, "Unknown")
    print(f"\n{result}")
    env.close()


# ========== Main ==========
if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, '.')
    from student_package.tron_env import BlindTronEnv
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train agent")
    parser.add_argument("--play", action="store_true", help="Play agent")
    parser.add_argument("--il-epochs", type=int, default=CONFIG['il_epochs'])
    parser.add_argument("--rl-episodes", type=int, default=CONFIG['rl_episodes'])
    parser.add_argument("--lr", type=float, default=CONFIG['lr'])
    args = parser.parse_args()
    
    CONFIG['il_epochs'] = args.il_epochs
    CONFIG['rl_episodes'] = args.rl_episodes
    CONFIG['lr'] = args.lr
    
    if args.train:
        train(BlindTronEnv, args.il_epochs, args.rl_episodes)
    elif args.play:
        play(BlindTronEnv)
    else:
        print("Hybrid IL + RL Agent")
        print("\nUsage:")
        print("  python student_rl.py --train --il-epochs 20 --rl-episodes 1000")
        print("  python student_rl.py --play")
