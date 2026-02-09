"""
RNN Tron Championship - Streamlit Cloud Version
支持学生自主上传和验证
"""

import streamlit as st
import numpy as np
import torch
from PIL import Image
import time
import os
import sys
from pathlib import Path
import json
from datetime import datetime

from tron_env import BlindTronEnv, EMPTY, WALL, P1_HEAD, P1_TRAIL, P2_HEAD, P2_TRAIL
from base_agent import RandomAgent, ExampleAgent
from tournament_runner import Tournament
from submission_manager import SubmissionManager

# 页面配置
st.set_page_config(
    page_title="RNN Tron Championship", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== 配置 ==========
SUBMISSIONS_DIR = Path("submissions")
SUBMISSIONS_DIR.mkdir(exist_ok=True)

# 颜色映射
COLORS = {
    EMPTY: [0, 0, 0],
    WALL: [100, 100, 100],
    P1_HEAD: [50, 50, 255],
    P1_TRAIL: [0, 0, 150],
    P2_HEAD: [255, 50, 50],
    P2_TRAIL: [150, 0, 0]
}

# ========== 工具函数 ==========

def grid_to_image(grid, cell_size=20):
    """转换网格为图片"""
    h, w = grid.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for val, color in COLORS.items():
        img[grid == val] = color
    return Image.fromarray(img).resize((w*cell_size, h*cell_size), Image.NEAREST)


@st.cache_resource
def load_all_agents():
    """加载所有Agent（内置 + 提交）"""
    agents = {"🎲 Random": RandomAgent(), "📚 Example": ExampleAgent()}
    manager = SubmissionManager(SUBMISSIONS_DIR)
    agents.update(manager.load_all_agents())
    return agents


def validate_submission(py_file, pth_file, student_name):
    """
    验证学生提交
    
    Returns:
        (is_valid, error_message, agent_instance)
    """
    try:
        # 保存上传的文件到临时位置
        temp_dir = SUBMISSIONS_DIR / f"temp_{student_name}"
        temp_dir.mkdir(exist_ok=True)
        
        py_path = temp_dir / f"{student_name}_agent.py"
        pth_path = temp_dir / f"{student_name}_agent.pth"
        
        with open(py_path, "wb") as f:
            f.write(py_file.getvalue())
        
        with open(pth_path, "wb") as f:
            f.write(pth_file.getvalue())
        
        # 尝试加载
        import importlib.util
        spec = importlib.util.spec_from_file_location(f"agent_{student_name}", py_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"agent_{student_name}"] = module
        spec.loader.exec_module(module)
        
        # 检查 StudentAgent 类
        if not hasattr(module, 'StudentAgent'):
            return False, "错误：找不到 StudentAgent 类", None
        
        agent = module.StudentAgent()
        
        # 加载权重
        state_dict = torch.load(pth_path, map_location='cpu')
        agent.load_state_dict(state_dict)
        agent.eval()
        
        # 检查参数量
        total_params = sum(p.numel() for p in agent.parameters())
        if total_params > 100_000:
            return False, f"错误：模型太大 ({total_params:,} 参数 > 100K 限制)", None
        
        # 测试推理
        dummy_obs = np.random.randn(10).astype(np.float32)
        agent.reset()
        action = agent.get_action(dummy_obs)
        
        if not isinstance(action, int) or action < 0 or action > 3:
            return False, f"错误：无效的动作输出 {action}", None
        
        # 验证通过，移动到正式目录
        final_py = SUBMISSIONS_DIR / f"{student_name}_agent.py"
        final_pth = SUBMISSIONS_DIR / f"{student_name}_agent.pth"
        
        # 如果已存在，备份旧版本
        if final_py.exists():
            backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_py.rename(SUBMISSIONS_DIR / f"{student_name}_agent_{backup_time}.py")
            final_pth.rename(SUBMISSIONS_DIR / f"{student_name}_agent_{backup_time}.pth")
        
        import shutil
        shutil.move(str(py_path), str(final_py))
        shutil.move(str(pth_path), str(final_pth))
        
        # 清理临时目录
        shutil.rmtree(temp_dir)
        
        # 记录提交日志
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "student": student_name,
            "params": total_params,
            "status": "success"
        }
        
        log_file = SUBMISSIONS_DIR / "submission_log.json"
        if log_file.exists():
            with open(log_file, "r") as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(log_entry)
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)
        
        return True, f"✅ 验证通过！模型参数: {total_params:,}", agent
        
    except Exception as e:
        return False, f"❌ 错误：{str(e)}", None


# ========== 页面布局 ==========

st.title("🐍 RNN Tron Championship")
st.markdown("---")

# 侧边栏 - 提交表单
with st.sidebar:
    st.header("📤 Submit Your Agent")
    
    with st.form("submission_form"):
        st.markdown("**Student Information**")
        student_name = st.text_input(
            "Name (English)", 
            placeholder="e.g., alice_wang",
            help="Will be used as your agent ID"
        )
        
        st.markdown("**Upload Files**")
        py_file = st.file_uploader(
            "Agent Code (.py)",
            type=['py'],
            help="Your student_template.py file"
        )
        
        pth_file = st.file_uploader(
            "Model Weights (.pth)",
            type=['pth'],
            help="Trained model weights"
        )
        
        submitted = st.form_submit_button("🚀 Submit & Validate", type="primary")
    
    if submitted:
        if not student_name or not py_file or not pth_file:
            st.error("❌ Please fill in all fields and upload both files!")
        else:
            # 清理姓名（只允许字母、数字、下划线）
            import re
            clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', student_name.lower())
            
            with st.spinner("Validating submission..."):
                is_valid, message, agent = validate_submission(
                    py_file, pth_file, clean_name
                )
            
            if is_valid:
                st.success(message)
                st.balloons()
                st.info("🎉 Your agent has been registered! Check the Leaderboard tab.")
                # 清除缓存，重新加载
                st.cache_resource.clear()
            else:
                st.error(message)
    
    st.divider()
    
    # 刷新按钮
    if st.button("🔄 Refresh All Agents"):
        st.cache_resource.clear()
        st.rerun()

# 加载所有Agent
agents = load_all_agents()

# 主内容区 - 标签页
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Tournament", 
    "🎮 Live Match", 
    "📊 Leaderboard",
    "📋 Submissions"
])

# ========== Tab 1: 锦标赛 ==========
with tab1:
    st.header("Round Robin Tournament")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        games = st.slider("Games per pair", 1, 5, 3)
        
        if st.button("🔥 Start Tournament", type="primary", use_container_width=True):
            if len(agents) >= 2:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("Running tournament..."):
                    tourney = Tournament()
                    
                    # 计算总比赛数
                    n = len(agents)
                    total_matches = n * (n - 1) // 2
                    current_match = 0
                    
                    # 手动运行以显示进度
                    names = list(agents.keys())
                    rankings_data = {name: {"points": 0, "wins": 0} for name in names}
                    
                    for i in range(len(names)):
                        for j in range(i + 1, len(names)):
                            current_match += 1
                            name1, name2 = names[i], names[j]
                            
                            status_text.text(f"Match {current_match}/{total_matches}: {name1} vs {name2}")
                            
                            wins = tourney.run_match(
                                agents[name1], agents[name2], 
                                name1, name2, games
                            )
                            
                            # 计分
                            if wins[name1] > wins[name2]:
                                rankings_data[name1]["points"] += 3
                                rankings_data[name1]["wins"] += 1
                            elif wins[name2] > wins[name1]:
                                rankings_data[name2]["points"] += 3
                                rankings_data[name2]["wins"] += 1
                            else:
                                rankings_data[name1]["points"] += 1
                                rankings_data[name2]["points"] += 1
                            
                            progress_bar.progress(current_match / total_matches)
                    
                    # 排序
                    rankings = sorted(
                        rankings_data.items(), 
                        key=lambda x: (-x[1]["points"], -x[1]["wins"])
                    )
                    
                    st.session_state['tournament_results'] = rankings
                    st.session_state['tournament_complete'] = True
                
                status_text.empty()
                progress_bar.empty()
                st.success("✅ Tournament complete!")
                
            else:
                st.error("Need at least 2 agents to start tournament")
    
    with col2:
        if st.session_state.get('tournament_complete'):
            st.subheader("Quick Results")
            rankings = st.session_state['tournament_results']
            
            for rank, (name, score) in enumerate(rankings[:5], 1):  # 只显示前5
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
                st.write(f"{medal} **{name}**: {score['points']} pts")
            
            if len(rankings) > 5:
                st.caption(f"... and {len(rankings) - 5} more agents")
        else:
            st.info("Click 'Start Tournament' to run the competition")

# ========== Tab 2: 观战 ==========
with tab2:
    st.header("Live Match")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        p1 = st.selectbox("Player 1 (Blue)", list(agents.keys()), 0)
    with col2:
        p2 = st.selectbox("Player 2 (Red)", list(agents.keys()), 
                         min(1, len(agents)-1))
    with col3:
        fps = st.slider("Speed (FPS)", 1, 30, 10)
    
    if p1 == p2:
        st.warning("Please select two different agents!")
    elif st.button("▶️ Play Match", use_container_width=True):
        env = BlindTronEnv(render_mode=False)
        obs1, obs2 = env.reset()
        agents[p1].reset()
        agents[p2].reset()
        
        placeholder = st.empty()
        done = False
        steps = 0
        
        while not done and steps < 400:
            with torch.no_grad():
                a1 = agents[p1].get_action(obs1)
                a2 = agents[p2].get_action(obs2)
            
            obs1, obs2, done, winner = env.step(a1, a2)
            
            img = grid_to_image(env.grid)
            placeholder.image(img, caption=f"Step {steps}", use_container_width=True)
            
            steps += 1
            time.sleep(1.0/fps)
        
        if winner == 1:
            st.success(f"🏆 {p1} wins!")
        elif winner == 2:
            st.success(f"🏆 {p2} wins!")
        else:
            st.info("🤝 Draw!")

# ========== Tab 3: 排行榜 ==========
with tab3:
    st.header("📊 Leaderboard")
    
    if st.session_state.get('tournament_complete'):
        rankings = st.session_state['tournament_results']
        
        # 统计卡片
        cols = st.columns(4)
        with cols[0]:
            st.metric("Total Agents", len(rankings))
        with cols[1]:
            top_score = rankings[0][1]['points'] if rankings else 0
            st.metric("Top Score", top_score)
        with cols[2]:
            matches = len(rankings) * (len(rankings) - 1) // 2
            st.metric("Matches", matches)
        with cols[3]:
            student_count = len([n for n, _ in rankings if n not in ["🎲 Random", "📚 Example"]])
            st.metric("Students", student_count)
        
        st.divider()
        
        # 完整排名表
        st.subheader("Full Rankings")
        
        # 表头
        cols = st.columns([1, 4, 2, 2, 2])
        cols[0].write("**Rank**")
        cols[1].write("**Agent**")
        cols[2].write("**Points**")
        cols[3].write("**Wins**")
        cols[4].write("**Status**")
        
        for rank, (name, score) in enumerate(rankings, 1):
            cols = st.columns([1, 4, 2, 2, 2])
            
            with cols[0]:
                if rank == 1:
                    st.write("🥇")
                elif rank == 2:
                    st.write("🥈")
                elif rank == 3:
                    st.write("🥉")
                else:
                    st.write(f"#{rank}")
            
            with cols[1]:
                st.write(f"**{name}**")
            
            with cols[2]:
                st.write(f"{score['points']}")
            
            with cols[3]:
                st.write(f"{score['wins']}")
            
            with cols[4]:
                if rank == 1:
                    st.caption("🏆 Champion!")
                elif score['points'] > 0:
                    st.caption("✓ Qualified")
    else:
        st.info("🎯 Run a tournament to see the rankings!")
        
        # 显示已注册Agent
        st.subheader("Registered Agents")
        for i, (name, agent) in enumerate(agents.items(), 1):
            cols = st.columns([1, 4, 3])
            with cols[0]:
                st.write(f"#{i}")
            with cols[1]:
                st.write(name)
            with cols[2]:
                try:
                    params = sum(p.numel() for p in agent.parameters())
                    st.caption(f"{params:,} params")
                except:
                    st.caption("Random agent")

# ========== Tab 4: 提交记录 ==========
with tab4:
    st.header("📋 Submission History")
    
    log_file = SUBMISSIONS_DIR / "submission_log.json"
    if log_file.exists():
        with open(log_file, "r") as f:
            logs = json.load(f)
        
        # 显示最近提交
        st.subheader("Recent Submissions")
        
        for log in reversed(logs[-10:]):  # 最近10条
            with st.container():
                cols = st.columns([2, 3, 2, 2])
                with cols[0]:
                    st.write(f"**{log['student']}**")
                with cols[1]:
                    st.caption(log['timestamp'][:19])
                with cols[2]:
                    st.write(f"{log['params']:,} params")
                with cols[3]:
                    status_emoji = "✅" if log['status'] == 'success' else "❌"
                    st.write(f"{status_emoji} {log['status']}")
        
        if len(logs) > 10:
            st.caption(f"... and {len(logs) - 10} more submissions")
    else:
        st.info("No submissions yet. Be the first!")
    
    # 显示所有文件
    st.divider()
    st.subheader("All Submission Files")
    
    py_files = list(SUBMISSIONS_DIR.glob("*_agent.py"))
    if py_files:
        for f in sorted(py_files):
            size = f.stat().st_size
            st.write(f"📄 {f.name} ({size:,} bytes)")
    else:
        st.caption("No .py files submitted yet")

# 页脚
st.markdown("---")
st.caption("RNN Tron Championship | Powered by Streamlit")
