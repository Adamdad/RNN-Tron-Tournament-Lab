"""
Better Data Generation for RNN Tron
====================================

生成多样化、高质量的专家数据用于模仿学习

Features:
- 多种专家策略 (Greedy, BFS, A*)
- 数据平衡 (确保动作分布均匀)
- 数据增强 (对称变换)
- 自适应探索

Usage:
    python generate_data.py --games 2000 --strategy bfs --balance --augment
    python generate_data.py --games 3000 --strategy mixed --verbose
"""

import numpy as np
import random
import argparse
from collections import deque, Counter
from typing import List, Tuple, Optional

from tron_env import BlindTronEnv, UP, DOWN, LEFT, RIGHT


class GreedyExpert:
    """贪心专家 - 选择不撞墙的安全动作"""
    
    def __init__(self, env, player_id=1, epsilon=0.1):
        self.env = env
        self.player_id = player_id
        self.epsilon = epsilon  # 探索概率
    
    def get_action(self, obs):
        pos = self.env.p1_pos if self.player_id == 1 else self.env.p2_pos
        curr_dir = self.env.p1_dir if self.player_id == 1 else self.env.p2_dir
        
        safe_actions = self._get_safe_actions(pos, curr_dir)
        
        if not safe_actions:
            return random.randint(0, 3)
        
        # Epsilon-greedy: 以 epsilon 概率随机探索
        if random.random() < self.epsilon:
            return random.choice(safe_actions)
        
        return random.choice(safe_actions)
    
    def _get_safe_actions(self, pos, curr_dir):
        """获取所有安全动作"""
        safe = []
        for action in [UP, DOWN, LEFT, RIGHT]:
            if not self._is_valid_turn(curr_dir, action):
                continue
            new_pos = self._move(pos, action)
            if not self._is_collision(new_pos):
                safe.append(action)
        return safe
    
    def _is_valid_turn(self, curr, new):
        if curr == -1:
            return True
        invalid = [(UP, DOWN), (DOWN, UP), (LEFT, RIGHT), (RIGHT, LEFT)]
        return (curr, new) not in invalid
    
    def _move(self, pos, action):
        r, c = pos
        if action == UP: r -= 1
        elif action == DOWN: r += 1
        elif action == LEFT: c -= 1
        elif action == RIGHT: c += 1
        return [r, c]
    
    def _is_collision(self, pos):
        r, c = pos
        if r < 0 or r >= self.env.grid_size or c < 0 or c >= self.env.grid_size:
            return True
        return self.env.grid[r, c] != 0


class BFSExpert:
    """BFS专家 - 选择能存活最久的方向"""
    
    def __init__(self, env, player_id=1, depth=30):
        self.env = env
        self.player_id = player_id
        self.depth = depth
    
    def get_action(self, obs):
        pos = self.env.p1_pos if self.player_id == 1 else self.env.p2_pos
        curr_dir = self.env.p1_dir if self.player_id == 1 else self.env.p2_dir
        
        best_action = None
        best_score = -1
        
        for action in [UP, DOWN, LEFT, RIGHT]:
            if not self._is_valid_turn(curr_dir, action):
                continue
            
            new_pos = self._move(pos, action)
            if self._is_collision(new_pos):
                continue
            
            # BFS计算从该方向能到达的空间
            score = self._bfs_score(new_pos)
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action if best_action is not None else random.randint(0, 3)
    
    def _bfs_score(self, start_pos):
        """BFS计算可达空间"""
        visited = set()
        queue = deque([(start_pos[0], start_pos[1])])
        visited.add((start_pos[0], start_pos[1]))
        count = 0
        
        while queue and count < self.depth:
            r, c = queue.popleft()
            count += 1
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in visited:
                    continue
                if nr < 0 or nr >= self.env.grid_size or nc < 0 or nc >= self.env.grid_size:
                    continue
                if self.env.grid[nr, nc] != 0:
                    continue
                visited.add((nr, nc))
                queue.append((nr, nc))
        
        return count
    
    def _is_valid_turn(self, curr, new):
        if curr == -1:
            return True
        invalid = [(UP, DOWN), (DOWN, UP), (LEFT, RIGHT), (RIGHT, LEFT)]
        return (curr, new) not in invalid
    
    def _move(self, pos, action):
        r, c = pos
        if action == UP: r -= 1
        elif action == DOWN: r += 1
        elif action == LEFT: c -= 1
        elif action == RIGHT: c += 1
        return [r, c]
    
    def _is_collision(self, pos):
        r, c = pos
        if r < 0 or r >= self.env.grid_size or c < 0 or c >= self.env.grid_size:
            return True
        return self.env.grid[r, c] != 0


class WallHuggerExpert:
    """贴墙专家 - 倾向于沿着墙壁移动（增加多样性）"""
    
    def __init__(self, env, player_id=1):
        self.env = env
        self.player_id = player_id
        self.greedy = GreedyExpert(env, player_id)
    
    def get_action(self, obs):
        pos = self.env.p1_pos if self.player_id == 1 else self.env.p2_pos
        curr_dir = self.env.p1_dir if self.player_id == 1 else self.env.p2_dir
        
        safe_actions = self.greedy._get_safe_actions(pos, curr_dir)
        if not safe_actions:
            return random.randint(0, 3)
        
        # 优先选择靠近墙壁的动作
        wall_scores = {}
        for action in safe_actions:
            new_pos = self._move(pos, action)
            wall_scores[action] = self._count_adjacent_walls(new_pos)
        
        # 选择墙最多的方向（有随机性）
        max_walls = max(wall_scores.values())
        best_actions = [a for a, s in wall_scores.items() if s == max_walls]
        return random.choice(best_actions)
    
    def _count_adjacent_walls(self, pos):
        """计算相邻的墙壁数"""
        count = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if nr < 0 or nr >= self.env.grid_size or nc < 0 or nc >= self.env.grid_size:
                count += 1
            elif self.env.grid[nr, nc] != 0:
                count += 1
        return count
    
    def _move(self, pos, action):
        r, c = pos
        if action == UP: r -= 1
        elif action == DOWN: r += 1
        elif action == LEFT: c -= 1
        elif action == RIGHT: c += 1
        return [r, c]


class MixedExpert:
    """混合专家 - 动态切换策略"""
    
    def __init__(self, env, player_id=1):
        self.env = env
        self.player_id = player_id
        self.experts = {
            'greedy': GreedyExpert(env, player_id, epsilon=0.2),
            'bfs': BFSExpert(env, player_id, depth=25),
            'wall': WallHuggerExpert(env, player_id)
        }
        self.current_expert = 'greedy'
        self.steps = 0
    
    def get_action(self, obs):
        self.steps += 1
        
        # 每50步切换一次策略
        if self.steps % 50 == 0:
            self.current_expert = random.choice(list(self.experts.keys()))
        
        return self.experts[self.current_expert].get_action(obs)


def balance_data(X, Y, max_ratio=2.0):
    """
    平衡数据集，确保动作分布均匀
    
    Args:
        X: 输入数据
        Y: 标签
        max_ratio: 最大允许的动作比例差异
    
    Returns:
        平衡后的 X, Y
    """
    action_counts = Counter(Y)
    min_count = min(action_counts.values())
    max_count = max(action_counts.values())
    
    if max_count / min_count <= max_ratio:
        # 已经足够平衡
        return X, Y
    
    print(f"  Balancing data: {dict(action_counts)}")
    
    # 确定每个动作的样本数上限
    max_per_action = int(min_count * max_ratio)
    
    balanced_X = []
    balanced_Y = []
    action_collected = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for x, y in zip(X, Y):
        if action_collected[y] < max_per_action:
            balanced_X.append(x)
            balanced_Y.append(y)
            action_collected[y] += 1
    
    # 如果某些动作太少，过采样
    for action in [0, 1, 2, 3]:
        while action_collected[action] < max_per_action * 0.8:
            # 找到该动作的所有样本并随机复制
            action_indices = [i for i, y in enumerate(Y) if y == action]
            if action_indices:
                idx = random.choice(action_indices)
                balanced_X.append(X[idx])
                balanced_Y.append(Y[idx])
                action_collected[action] += 1
    
    return np.array(balanced_X), np.array(balanced_Y)


def augment_data(X, Y):
    """
    数据增强：水平翻转
    
    观测顺序: [N, NE, E, SE, S, SW, W, NW, x, y]
    水平翻转后: [N, NW, W, SW, S, SE, E, NE, x, y]
    """
    X_aug = []
    Y_aug = []
    
    for x, y in zip(X, Y):
        # 原样本
        X_aug.append(x)
        Y_aug.append(y)
        
        # 水平翻转
        x_flip = x.copy()
        # 交换 E<->W, NE<->NW, SE<->SW
        x_flip[:, [2, 6]] = x_flip[:, [6, 2]]  # E <-> W
        x_flip[:, [1, 7]] = x_flip[:, [7, 1]]  # NE <-> NW
        x_flip[:, [3, 5]] = x_flip[:, [5, 3]]  # SE <-> SW
        
        # 动作翻转: LEFT<->RIGHT, UP/DOWN不变
        flip_map = {UP: UP, DOWN: DOWN, LEFT: RIGHT, RIGHT: LEFT}
        y_flip = flip_map[y]
        
        X_aug.append(x_flip)
        Y_aug.append(y_flip)
    
    return np.array(X_aug), np.array(Y_aug)


def generate_dataset(
    num_games=1000,
    seq_len=10,
    strategy='mixed',
    balance=False,
    augment=False,
    min_game_length=20,
    verbose=True
):
    """
    生成训练数据集
    
    Args:
        num_games: 游戏局数
        seq_len: 序列长度
        strategy: 策略 ('greedy', 'bfs', 'wall', 'mixed')
        balance: 是否平衡数据
        augment: 是否数据增强
        min_game_length: 最小游戏长度（过滤太短的游戏）
        verbose: 是否打印详细信息
    """
    env = BlindTronEnv(grid_size=20, render_mode=False)
    
    X, Y = [], []
    game_lengths = []
    action_counts = [0, 0, 0, 0]
    
    if verbose:
        print(f"Generating {num_games} games with '{strategy}' strategy...")
    
    for game in range(num_games):
        if verbose and (game + 1) % 100 == 0:
            print(f"  Progress: {game + 1}/{num_games}")
        
        # 创建Agent
        if strategy == 'greedy':
            agent1 = GreedyExpert(env, 1, epsilon=0.15)
            agent2 = GreedyExpert(env, 2, epsilon=0.15)
        elif strategy == 'bfs':
            agent1 = BFSExpert(env, 1)
            agent2 = BFSExpert(env, 2)
        elif strategy == 'wall':
            agent1 = WallHuggerExpert(env, 1)
            agent2 = WallHuggerExpert(env, 2)
        elif strategy == 'mixed':
            agent1 = MixedExpert(env, 1)
            agent2 = MixedExpert(env, 2)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        obs1, obs2 = env.reset()
        history1 = []
        done = False
        steps = 0
        max_steps = 400
        
        while not done and steps < max_steps:
            action1 = agent1.get_action(obs1)
            action2 = agent2.get_action(obs2)
            
            history1.append(obs1)
            
            if len(history1) >= seq_len:
                X.append(np.array(history1[-seq_len:]))
                Y.append(action1)
                action_counts[action1] += 1
            
            obs1, obs2, done, winner = env.step(action1, action2)
            steps += 1
        
        if steps >= min_game_length:
            game_lengths.append(steps)
    
    X = np.array(X)
    Y = np.array(Y)
    
    if verbose:
        print(f"\nGenerated {len(X)} samples from {len(game_lengths)} games")
        print(f"Average game length: {np.mean(game_lengths):.1f} steps")
        print(f"Action distribution: {action_counts}")
    
    # 数据平衡
    if balance:
        X, Y = balance_data(X, Y)
        if verbose:
            new_counts = Counter(Y)
            print(f"After balancing: {dict(new_counts)}")
    
    # 数据增强
    if augment:
        X, Y = augment_data(X, Y)
        if verbose:
            print(f"After augmentation: {len(X)} samples")
    
    return X, Y


def split_dataset(X, Y, train_ratio=0.9):
    """分割训练集和验证集"""
    n = len(X)
    indices = np.random.permutation(n)
    train_size = int(n * train_ratio)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]
    
    return X[train_idx], Y[train_idx], X[val_idx], Y[val_idx]


def analyze_dataset(X_path="train_X.npy", Y_path="train_Y.npy"):
    """分析数据集"""
    X = np.load(X_path)
    Y = np.load(Y_path)
    
    print("\nDataset Analysis:")
    print(f"  Shape: X={X.shape}, Y={Y.shape}")
    print(f"  Dtype: X={X.dtype}, Y={Y.dtype}")
    print(f"  X range: [{X.min():.3f}, {X.max():.3f}]")
    
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    counts = Counter(Y)
    print(f"\nAction distribution:")
    for action, count in sorted(counts.items()):
        pct = 100 * count / len(Y)
        print(f"  {action_names[action]}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate training data for RNN Tron')
    parser.add_argument('--games', type=int, default=1000, help='Number of games')
    parser.add_argument('--seq-len', type=int, default=10, help='Sequence length')
    parser.add_argument('--strategy', type=str, default='mixed',
                       choices=['greedy', 'bfs', 'wall', 'mixed'],
                       help='Expert strategy')
    parser.add_argument('--balance', action='store_true', help='Balance action distribution')
    parser.add_argument('--augment', action='store_true', help='Data augmentation')
    parser.add_argument('--split', type=float, default=None, help='Train/val split ratio')
    parser.add_argument('--analyze', action='store_true', help='Analyze existing dataset')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_dataset()
    else:
        print("=" * 60)
        print("RNN Tron Data Generator")
        print("=" * 60)
        
        X, Y = generate_dataset(
            num_games=args.games,
            seq_len=args.seq_len,
            strategy=args.strategy,
            balance=args.balance,
            augment=args.augment
        )
        
        # 保存
        if args.split:
            X_train, Y_train, X_val, Y_val = split_dataset(X, Y, args.split)
            np.save("train_X.npy", X_train)
            np.save("train_Y.npy", Y_train)
            np.save("val_X.npy", X_val)
            np.save("val_Y.npy", Y_val)
            print(f"\nSaved: train ({len(X_train)}), val ({len(X_val)})")
        else:
            np.save("train_X.npy", X)
            np.save("train_Y.npy", Y)
            print(f"\nSaved: train_X.npy, train_Y.npy ({len(X)} samples)")
        
        print("\n✓ Data generation complete!")
