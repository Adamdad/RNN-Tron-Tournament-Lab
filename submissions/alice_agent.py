"""
学生提交模板 - 实现你的RNN Agent
"""

import torch
import torch.nn as nn
import numpy as np

# ========== 填写你的信息 ==========
STUDENT_INFO = {
    "name": "Alice Wang",           # 姓名
    "student_id": "2024001001",     # 学号
    "team_name": "闪电队",          # 队伍名
    "description": "双层LSTM"       # 简短描述
}


# ========== 定义你的模型 ==========
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: 定义你的网络结构
        self.lstm = nn.LSTM(10, 64, batch_first=True)
        self.fc = nn.Linear(64, 4)
    
    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        return self.fc(out[:, -1, :]), hidden


# ========== Agent类（不要修改类名）==========
class StudentAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = STUDENT_INFO.get("team_name") or STUDENT_INFO["name"]
        self.info = STUDENT_INFO
        self.model = MyModel()
        self.hidden = None
    
    def reset(self):
        self.hidden = None
    
    def get_action(self, obs):
        x = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            if self.hidden is not None:
                out, self.hidden = self.model(x, self.hidden)
            else:
                out = self.model(x)
                if isinstance(out, tuple):
                    out, self.hidden = out
            return torch.argmax(out, dim=-1).item()
    
    def forward(self, x, hidden=None):
        return self.model(x, hidden)


# ========== 训练代码 ==========
def train(data_path="train_X.npy", labels_path="train_Y.npy", 
          epochs=10, lr=0.001, batch_size=32):
    """训练模型"""
    print("Loading data...")
    X = np.load(data_path)
    Y = np.load(labels_path)
    print(f"Loaded {len(X)} samples")
    
    agent = StudentAgent()
    print(f"Model has {sum(p.numel() for p in agent.parameters())} params")
    
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X), torch.LongTensor(Y)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = agent.model(batch_x)
            if isinstance(out, tuple):
                out = out[0]
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (torch.argmax(out, dim=1) == batch_y).sum().item()
        
        acc = 100 * correct / len(X)
        print(f"Epoch {epoch+1}/{epochs}: Loss={total_loss/len(loader):.4f}, Acc={acc:.2f}%")
    
    # 保存模型
    import os
    os.makedirs("submissions", exist_ok=True)
    save_path = f"submissions/{STUDENT_INFO['name'].replace(' ', '_').lower()}_agent.pth"
    torch.save(agent.state_dict(), save_path)
    print(f"\nSaved to {save_path}")
    return agent


# ========== 主程序 ==========
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    
    if args.train:
        train(epochs=args.epochs, lr=args.lr)
    else:
        print("Student Agent Template")
        print("Usage: python student_template.py --train")
