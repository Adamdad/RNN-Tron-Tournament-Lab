# RNN Tron Tournament Lab

学生训练 RNN Agent 对战 Tron 游戏的锦标赛系统。

## 文件结构

```
lab/
├── base_agent.py           # Agent基类（68行）
├── student_template.py     # 学生模板（95行，含训练代码）
├── generate_data.py        # 生成专家数据（93行）
├── tournament_runner.py    # 锦标赛运行器（92行）
├── tournament_app.py       # Streamlit界面（116行）
├── submission_manager.py   # 提交管理器（54行）
├── tron_env.py            # 游戏环境（保留）
└── submissions/           # 学生提交目录
```

## 使用流程

### 1. 教师生成数据

```bash
python generate_data.py --games 1000
```

### 2. 学生开发

复制模板并修改：

```bash
cp student_template.py submissions/your_name_agent.py
```

修改 `STUDENT_INFO` 和 `MyModel` 类，然后训练：

```bash
python submissions/your_name_agent.py --train --epochs 20
```

### 3. 运行锦标赛

```bash
# 命令行
python tournament_runner.py

# 或 Streamlit 界面
streamlit run tournament_app.py
```

## 核心类说明

### StudentAgent (学生必须实现)

```python
class StudentAgent(nn.Module):
    def __init__(self):
        self.name = "TeamName"
        self.model = MyModel()  # 你的RNN模型
        self.hidden = None
    
    def reset(self):
        """每局游戏开始前调用"""
        self.hidden = None
    
    def get_action(self, obs):
        """根据观测选择动作 (0=上, 1=下, 2=左, 3=右)"""
        # obs: numpy array (10,)
        # 返回: int 动作
```

## 简化说明

- **base_agent.py**: 从 202 行 → 68 行
- **student_template.py**: 从 470 行 → 95 行  
- **generate_data.py**: 从 504 行 → 93 行
- **tournament_runner.py**: 从 312 行 → 92 行
- **tournament_app.py**: 从 315 行 → 116 行
- **submission_manager.py**: 从 330 行 → 54 行

总计从 2133 行简化到 518 行（减少 75%）

## 依赖

```bash
pip install torch numpy pygame streamlit pillow
```
