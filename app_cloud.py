"""
RNN Tron Championship - Streamlit Cloud Version
Supports student self-upload and validation
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

# Page config
st.set_page_config(
    page_title="RNN Tron Championship", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== Configuration ==========
SUBMISSIONS_DIR = Path("submissions")
SUBMISSIONS_DIR.mkdir(exist_ok=True)

# Color mapping
COLORS = {
    EMPTY: [0, 0, 0],
    WALL: [100, 100, 100],
    P1_HEAD: [50, 50, 255],
    P1_TRAIL: [0, 0, 150],
    P2_HEAD: [255, 50, 50],
    P2_TRAIL: [150, 0, 0]
}

# ========== Utility Functions ==========

def grid_to_image(grid, cell_size=12):
    """Convert grid to image with smaller default size for better fit"""
    h, w = grid.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for val, color in COLORS.items():
        img[grid == val] = color
    return Image.fromarray(img).resize((w*cell_size, h*cell_size), Image.NEAREST)


@st.cache_resource
def load_all_agents():
    """Load all Agents (built-in + submitted)"""
    agents = {"🎲 Random": RandomAgent(), "📚 Example": ExampleAgent()}
    manager = SubmissionManager(str(SUBMISSIONS_DIR))
    agents.update(manager.load_all_agents())
    return agents


def load_tournament_results():
    """Load saved tournament results from file"""
    results_file = SUBMISSIONS_DIR / "tournament_results.json"
    if results_file.exists():
        try:
            with open(results_file, "r") as f:
                data = json.load(f)
                # Convert back to list of tuples with proper score dict format
                rankings = []
                for item in data:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        name, scores = item
                        # Ensure scores is a dict with required keys
                        if isinstance(scores, dict) and 'points' in scores and 'wins' in scores:
                            rankings.append((name, scores))
                return rankings if rankings else None
        except Exception as e:
            print(f"Error loading tournament results: {e}")
            return None
    return None


def save_tournament_results(rankings):
    """Save tournament results to file"""
    results_file = SUBMISSIONS_DIR / "tournament_results.json"
    try:
        with open(results_file, "w") as f:
            json.dump(rankings, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving tournament results: {e}")
        return False


def delete_submission_files(name):
    """Delete submission files directly"""
    try:
        py_file = SUBMISSIONS_DIR / f"{name}_agent.py"
        pth_file = SUBMISSIONS_DIR / f"{name}_agent.pth"
        
        deleted = False
        
        if py_file.exists():
            py_file.unlink()
            deleted = True
        
        if pth_file.exists():
            pth_file.unlink()
            deleted = True
        
        # Delete backups
        for backup_file in SUBMISSIONS_DIR.glob(f"{name}_agent_*.py"):
            backup_file.unlink()
        
        for backup_file in SUBMISSIONS_DIR.glob(f"{name}_agent_*.pth"):
            backup_file.unlink()
        
        return deleted
    except Exception as e:
        st.error(f"Error deleting files: {e}")
        return False


def validate_submission(py_file, pth_file, student_name):
    """
    Validate student submission
    
    Returns:
        (is_valid, error_message, agent_instance)
    """
    try:
        # Save uploaded files to temporary location
        temp_dir = SUBMISSIONS_DIR / f"temp_{student_name}"
        temp_dir.mkdir(exist_ok=True)
        
        py_path = temp_dir / f"{student_name}_agent.py"
        pth_path = temp_dir / f"{student_name}_agent.pth"
        
        with open(py_path, "wb") as f:
            f.write(py_file.getvalue())
        
        with open(pth_path, "wb") as f:
            f.write(pth_file.getvalue())
        
        # Try to load
        import importlib.util
        spec = importlib.util.spec_from_file_location(f"agent_{student_name}", py_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"agent_{student_name}"] = module
        spec.loader.exec_module(module)
        
        # Check for StudentAgent class
        if not hasattr(module, 'StudentAgent'):
            return False, "Error: StudentAgent class not found", None
        
        agent = module.StudentAgent()
        
        # Load weights
        state_dict = torch.load(pth_path, map_location='cpu')
        agent.load_state_dict(state_dict)
        agent.eval()
        
        # Check parameter count
        total_params = sum(p.numel() for p in agent.parameters())
        if total_params > 100_000:
            return False, f"Error: Model too large ({total_params:,} params > 100K limit)", None
        
        # Test inference
        dummy_obs = np.random.randn(10).astype(np.float32)
        agent.reset()
        action = agent.get_action(dummy_obs)
        
        if not isinstance(action, int) or action < 0 or action > 3:
            return False, f"Error: Invalid action output {action}", None
        
        # Validation passed, move to official directory
        final_py = SUBMISSIONS_DIR / f"{student_name}_agent.py"
        final_pth = SUBMISSIONS_DIR / f"{student_name}_agent.pth"
        
        # Backup old version if exists
        if final_py.exists():
            backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_py.rename(SUBMISSIONS_DIR / f"{student_name}_agent_{backup_time}.py")
            final_pth.rename(SUBMISSIONS_DIR / f"{student_name}_agent_{backup_time}.pth")
        
        import shutil
        shutil.move(str(py_path), str(final_py))
        shutil.move(str(pth_path), str(final_pth))
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        
        # Log submission
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
        
        return True, f"✅ Validation passed! Model parameters: {total_params:,}", agent
        
    except Exception as e:
        return False, f"❌ Error: {str(e)}", None


# ========== Page Layout ==========

st.title("🐍 RNN Tron Championship")
st.markdown("---")

# Sidebar - submission form
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
            # Clean name (only letters, numbers, underscores)
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
                # Clear cache and reload
                st.cache_resource.clear()
            else:
                st.error(message)
    
    st.divider()
    
    # Refresh button
    if st.button("🔄 Refresh All Agents"):
        st.cache_resource.clear()
        st.rerun()

# Load all Agents
agents = load_all_agents()

# Load saved tournament results on startup
saved_results = load_tournament_results()
if saved_results and not st.session_state.get('tournament_results'):
    st.session_state['tournament_results'] = saved_results
    st.session_state['tournament_complete'] = True

# Main content area - tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Tournament", 
    "🎮 Live Match", 
    "📊 Leaderboard",
    "📋 Submissions"
])

# ========== Tab 1: Tournament ==========
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
                    
                    # Calculate total matches
                    n = len(agents)
                    total_matches = n * (n - 1) // 2
                    current_match = 0
                    
                    # Manual run to show progress
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
                            
                            # Scoring
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
                    
                    # Sort
                    rankings = sorted(
                        rankings_data.items(), 
                        key=lambda x: (-x[1]["points"], -x[1]["wins"])
                    )
                    
                    st.session_state['tournament_results'] = rankings
                    st.session_state['tournament_complete'] = True
                    st.session_state['tournament_timestamp'] = datetime.now().isoformat()
                    
                    # Save to file
                    if save_tournament_results(rankings):
                        st.success("✅ Tournament complete! Results saved.")
                    else:
                        st.warning("✅ Tournament complete! But failed to save results.")
                
                status_text.empty()
                progress_bar.empty()
                
            else:
                st.error("Need at least 2 agents to start tournament")
    
    with col2:
        if st.session_state.get('tournament_complete'):
            st.subheader("Quick Results")
            
            # Show timestamp if available
            if 'tournament_timestamp' in st.session_state:
                timestamp = st.session_state['tournament_timestamp']
                try:
                    dt = datetime.fromisoformat(timestamp)
                    st.caption(f"Last run: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                except:
                    pass
            
            rankings = st.session_state['tournament_results']
            
            for rank, (name, score) in enumerate(rankings[:5], 1):  # Show top 5 only
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
                st.write(f"{medal} **{name}**: {score['points']} pts")
            
            if len(rankings) > 5:
                st.caption(f"... and {len(rankings) - 5} more agents")
                
            # Add clear results button
            st.divider()
            if st.button("🗑️ Clear Saved Results"):
                results_file = SUBMISSIONS_DIR / "tournament_results.json"
                if results_file.exists():
                    results_file.unlink()
                st.session_state['tournament_results'] = None
                st.session_state['tournament_complete'] = False
                st.session_state.pop('tournament_timestamp', None)
                st.success("Results cleared!")
                st.rerun()
        else:
            st.info("Click 'Start Tournament' to run the competition")

# ========== Tab 2: Live Match ==========
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
    else:
        # Run game button
        if st.button("▶️ Play Match", use_container_width=True, key="play_match"):
            with st.spinner("Running game..."):
                # Initialize
                env = BlindTronEnv(render_mode=False)
                obs1, obs2 = env.reset()
                agents[p1].reset()
                agents[p2].reset()
                
                # Create placeholder for visualization
                placeholder = st.empty()
                
                # Run game loop
                done = False
                steps = 0
                winner = 0
                
                while not done and steps < 400:
                    with torch.no_grad():
                        a1 = agents[p1].get_action(obs1)
                        a2 = agents[p2].get_action(obs2)
                    
                    obs1, obs2, done, winner = env.step(a1, a2)
                    
                    # Update display
                    img = grid_to_image(env.grid)
                    placeholder.image(img, caption=f"Step {steps} | 🔵 {p1} vs 🔴 {p2}", width=240)
                    
                    steps += 1
                    time.sleep(1.0/fps)
                
                # Show result
                if winner == 1:
                    st.success(f"🏆 {p1} wins!")
                elif winner == 2:
                    st.success(f"🏆 {p2} wins!")
                else:
                    st.info("🤝 Draw!")

# ========== Tab 3: Leaderboard ==========
with tab3:
    st.header("📊 Leaderboard")
    
    # Show timestamp if available
    if st.session_state.get('tournament_timestamp'):
        timestamp = st.session_state['tournament_timestamp']
        try:
            dt = datetime.fromisoformat(timestamp)
            st.caption(f"Results from: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        except:
            pass
    
    if st.session_state.get('tournament_complete'):
        rankings = st.session_state['tournament_results']
        
        # Stat cards
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
        
        # Full rankings table
        st.subheader("Full Rankings")
        
        # Headers
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
        
        # Show registered agents
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

# ========== Tab 4: Submission History ==========
with tab4:
    st.header("📋 Submission History")
    
    # Get list of submissions directly from filesystem
    py_files = list(SUBMISSIONS_DIR.glob("*_agent.py"))
    submissions = []
    for f in py_files:
        name = f.stem.replace("_agent", "")
        submissions.append(name)
    submissions = sorted(submissions)
    
    # Delete Submission Section
    st.subheader("🗑️ Manage Submissions")
    
    if submissions:
        col1, col2 = st.columns([3, 1])
        with col1:
            submission_to_delete = st.selectbox(
                "Select submission to delete",
                submissions,
                key="delete_select"
            )
        with col2:
            st.write("")
            st.write("")
            if st.button("🗑️ Delete", type="secondary", use_container_width=True):
                if submission_to_delete:
                    with st.spinner(f"Deleting {submission_to_delete}..."):
                        success = delete_submission_files(submission_to_delete)
                        if success:
                            # Also remove from log
                            log_file = SUBMISSIONS_DIR / "submission_log.json"
                            if log_file.exists():
                                with open(log_file, "r") as f:
                                    logs = json.load(f)
                                # Filter out logs for this student
                                logs = [log for log in logs if log['student'] != submission_to_delete]
                                with open(log_file, "w") as f:
                                    json.dump(logs, f, indent=2)
                            
                            st.success(f"✅ Deleted {submission_to_delete}")
                            st.cache_resource.clear()
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to delete {submission_to_delete}")
        
        # Delete all button
        st.divider()
        col1, col2, col3 = st.columns([2, 2, 2])
        with col2:
            if st.button("🗑️ Delete ALL Submissions", type="primary", use_container_width=True):
                confirm_col1, confirm_col2 = st.columns(2)
                with confirm_col1:
                    if st.button("✅ Confirm Delete All", type="secondary", use_container_width=True):
                        with st.spinner("Deleting all submissions..."):
                            deleted_count = 0
                            for sub in submissions:
                                if delete_submission_files(sub):
                                    deleted_count += 1
                            
                            # Clear log file
                            log_file = SUBMISSIONS_DIR / "submission_log.json"
                            if log_file.exists():
                                log_file.unlink()
                            
                            # Clear tournament results
                            results_file = SUBMISSIONS_DIR / "tournament_results.json"
                            if results_file.exists():
                                results_file.unlink()
                            
                            st.success(f"✅ Deleted {deleted_count} submissions")
                            st.cache_resource.clear()
                            time.sleep(1)
                            st.rerun()
                with confirm_col2:
                    if st.button("❌ Cancel", use_container_width=True):
                        st.rerun()
    else:
        st.info("No submissions to delete")
    
    st.divider()
    
    # Show recent submissions
    log_file = SUBMISSIONS_DIR / "submission_log.json"
    if log_file.exists():
        with open(log_file, "r") as f:
            logs = json.load(f)
        
        st.subheader("Recent Submissions")
        
        for log in reversed(logs[-10:]):  # Last 10
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
        st.info("No submission history yet")
    
    # Show all files
    st.divider()
    st.subheader("All Submission Files")
    
    if py_files:
        for f in sorted(py_files):
            size = f.stat().st_size
            st.write(f"📄 {f.name} ({size:,} bytes)")
    else:
        st.caption("No .py files submitted yet")

# Footer
st.markdown("---")
st.caption("RNN Tron Championship | Powered by Streamlit")
