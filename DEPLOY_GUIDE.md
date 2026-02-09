# RNN Tron Challenge - Streamlit Cloud 部署指南

## 🚀 快速部署（5分钟搞定）

### 步骤 1: 准备 GitHub 仓库

```bash
# 1. 创建新的 GitHub 仓库（例如：tron-challenge）
# 2. 复制 lab/ 文件夹内容到仓库根目录

# 目录结构应该是：
tron-challenge/
├── app_cloud.py           # 主应用（刚创建的）
├── base_agent.py
├── student_template.py
├── generate_data.py
├── tournament_runner.py
├── submission_manager.py
├── tron_env.py
├── requirements.txt       # 需要创建
├── packages.txt          # 系统依赖（可选）
└── submissions/          # 空文件夹
    └── .gitkeep
```

### 步骤 2: 创建 requirements.txt

```txt
streamlit>=1.28.0
torch>=2.0.0
numpy>=1.24.0
pygame>=2.5.0
pillow>=10.0.0
```

### 步骤 3: 创建 packages.txt（系统依赖）

```txt
# 用于 pygame
libsdl2-dev
libsdl2-image-dev
libsdl2-mixer-dev
libsdl2-ttf-dev
```

### 步骤 4: 部署到 Streamlit Cloud

1. 访问 [streamlit.io/cloud](https://streamlit.io/cloud)
2. 点击 "New app"
3. 选择你的 GitHub 仓库
4. 配置文件选择 `app_cloud.py`
5. 点击 "Deploy"

**等待 2-3 分钟，应用就会自动上线！**

---

## 📋 学生使用流程

### 1. 开发模型（本地）

```bash
# 学生本地训练
python student_template.py --train --epochs 20
```

### 2. 提交到网站

1. 打开 Streamlit 应用链接
2. 在侧边栏填写姓名
3. 上传两个文件：
   - `alice_agent.py`（代码）
   - `alice_agent.pth`（权重）
4. 点击 "Submit & Validate"
5. 系统自动验证并显示结果

### 3. 查看排名

- 进入 "Leaderboard" 标签页
- 点击 "Start Tournament" 运行比赛
- 查看实时排名

---

## 🔧 高级配置

### 持久化存储（重要！）

Streamlit Cloud 重启后文件会丢失。有两种解决方案：

#### 方案 A: 使用外部存储（推荐）

修改 `app_cloud.py` 中的存储路径：

```python
# 使用 AWS S3
import boto3
s3 = boto3.client('s3')
BUCKET_NAME = 'your-bucket'

# 或者使用 Google Drive
# 或者使用数据库（Supabase, Firebase等）
```

#### 方案 B: 定期备份到 GitHub

添加自动备份功能：

```python
# 在 app_cloud.py 中添加
import subprocess

def backup_to_github():
    subprocess.run(['git', 'add', 'submissions/'])
    subprocess.run(['git', 'commit', '-m', 'Auto backup'])
    subprocess.run(['git', 'push'])
```

#### 方案 C: 使用 Streamlit 的持久化（最简单）

```python
# 使用 st.session_state 和 @st.cache_resource
# 但文件仍然只在内存中
```

**推荐：对于课堂使用，方案 A 最可靠。**

---

## 🎨 自定义配置

### 修改页面标题

在 `app_cloud.py` 中修改：

```python
st.set_page_config(
    page_title="你的课程名称 - RNN Challenge", 
    layout="wide"
)
```

### 修改评分规则

在 `tournament_runner.py` 中修改计分：

```python
# 当前：胜3分，平1分，负0分
# 可以改为其他规则
```

### 添加参数限制

在 `validate_submission()` 函数中：

```python
# 当前限制：100K 参数
if total_params > 100_000:
    return False, "Too many parameters", None

# 可以添加其他限制
if inference_time > 0.05:  # 50ms
    return False, "Too slow", None
```

---

## 📊 监控和统计

### 查看提交日志

在 "Submissions" 标签页可以看到：
- 提交时间
- 学生姓名
- 模型参数量
- 验证状态

### 导出结果

在 Leaderboard 页面可以添加导出按钮：

```python
if st.button("📥 Export Results"):
    rankings = st.session_state['tournament_results']
    import pandas as pd
    df = pd.DataFrame(rankings)
    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        "leaderboard.csv"
    )
```

---

## 🐛 常见问题

### Q: 文件上传后消失？
A: Streamlit Cloud 重启后文件会丢失。使用外部存储或定期备份。

### Q: 模型加载失败？
A: 检查：
1. 类名是否为 `StudentAgent`
2. 模型结构是否与保存时一致
3. 文件是否完整上传

### Q: 并发用户限制？
A: Streamlit Cloud 免费版支持同时约 10-20 个用户。如需更多，升级 Team 版。

### Q: 如何重置所有提交？
A: 在侧边栏点击 "Refresh All Agents"，或在 GitHub 中清空 `submissions/` 文件夹。

---

## 💡 教学建议

### 课程流程

1. **Week 1**: 发布挑战，讲解 RNN 和游戏规则
2. **Week 2-3**: 学生本地开发，提交到网站
3. **Week 4**: 运行最终锦标赛，公布排名
4. **Week 5**: 获胜者分享经验

### 防止作弊

1. **代码审查**: 定期检查提交代码
2. **版本控制**: 要求提交训练日志
3. **限时提交**: 设置截止日期后锁定上传
4. **多样性检查**: 检测相似模型（使用模型哈希）

### 激励措施

- 🏆 排行榜前3名获得加分
- 🎁 最佳创新奖（如使用 Transformer）
- 📚 代码质量奖（最佳注释和文档）

---

## 🔗 相关链接

- Streamlit Cloud: https://streamlit.io/cloud
- 文档: https://docs.streamlit.io/
- 社区: https://discuss.streamlit.io/

---

**祝教学愉快！🎉**
