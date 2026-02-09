"""
Streamlit Tournament Interface
"""

import streamlit as st
import numpy as np
import torch
from PIL import Image
import time

from tron_env import BlindTronEnv, EMPTY, WALL, P1_HEAD, P1_TRAIL, P2_HEAD, P2_TRAIL
from base_agent import RandomAgent, ExampleAgent
from tournament_runner import Tournament
from submission_manager import SubmissionManager

st.set_page_config(page_title="RNN Tron Championship", layout="wide")

# Color mapping
COLORS = {
    EMPTY: [0, 0, 0],
    WALL: [100, 100, 100],
    P1_HEAD: [50, 50, 255],
    P1_TRAIL: [0, 0, 150],
    P2_HEAD: [255, 50, 50],
    P2_TRAIL: [150, 0, 0]
}


def grid_to_image(grid, cell_size=20):
    """Convert grid to image"""
    h, w = grid.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for val, color in COLORS.items():
        img[grid == val] = color
    return Image.fromarray(img).resize((w*cell_size, h*cell_size), Image.NEAREST)


@st.cache_resource
def load_agents():
    """Load Agents"""
    agents = {"🎲 Random": RandomAgent(), "📚 Example": ExampleAgent()}
    manager = SubmissionManager("submissions")
    agents.update(manager.load_all_agents())
    return agents


# Page layout
st.title("🐍 RNN Tron Championship")

agents = load_agents()
st.sidebar.success(f"Loaded {len(agents)} agents")

# Tabs
tab1, tab2, tab3 = st.tabs(["🏆 Tournament", "🎮 Live Match", "📊 Leaderboard"])

# === Tournament ===
with tab1:
    st.header("Round Robin Tournament")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        games = st.slider("Games per pair", 1, 5, 3)
        if st.button("🔥 Start Tournament", type="primary"):
            if len(agents) >= 2:
                with st.spinner("Running tournament..."):
                    tourney = Tournament()
                    rankings = tourney.run_tournament(agents, games)
                    
                    # Save results to session state
                    st.session_state['tournament_results'] = rankings
                    st.session_state['tournament_complete'] = True
                
                # Show completion message
                st.success(f"✅ Tournament complete! Go to Leaderboard tab to see rankings.")
            else:
                st.error("Need at least 2 agents")
    
    with col2:
        if st.session_state.get('tournament_complete'):
            st.info("✓ Tournament results available in Leaderboard tab")
        else:
            st.info("Click 'Start Tournament' to run round-robin competition")

# === Live Match ===
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
    
    if st.button("▶️ Play") and p1 != p2:
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
            placeholder.image(img, caption=f"Step {steps}")
            
            steps += 1
            time.sleep(1.0/fps)
        
        if winner == 1:
            st.success(f"🏆 {p1} wins!")
        elif winner == 2:
            st.success(f"🏆 {p2} wins!")
        else:
            st.info("🤝 Draw!")

# === Leaderboard ===
with tab3:
    st.header("📊 Leaderboard")
    
    if st.session_state.get('tournament_complete'):
        # Show tournament results
        rankings = st.session_state['tournament_results']
        
        st.success(f"🏆 Tournament Results ({len(rankings)} agents)")
        
        # Headers
        cols = st.columns([1, 4, 2, 2, 2])
        cols[0].write("**Rank**")
        cols[1].write("**Agent**")
        cols[2].write("**Points**")
        cols[3].write("**Wins**")
        cols[4].write("**Status**")
        
        st.divider()
        
        # Show rankings
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
                st.write(f"{score['points']} pts")
            
            with cols[3]:
                st.write(f"{score['wins']} wins")
            
            with cols[4]:
                if rank == 1:
                    st.caption("🎉 Champion!")
                elif score['points'] > 0:
                    st.caption("✓ Qualified")
                else:
                    st.caption("-")
        
        st.divider()
        
        # Show statistics
        total_games = sum(score['wins'] for _, score in rankings) * 3  # Estimate
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Agents", len(rankings))
        with col2:
            top_score = rankings[0][1]['points'] if rankings else 0
            st.metric("Top Score", top_score)
        with col3:
            st.metric("Matches Played", f"{len(rankings) * (len(rankings)-1) // 2}")
    
    else:
        # No tournament run yet, show registered list
        st.info("🏃 No tournament results yet. Run a tournament in the 'Tournament' tab first.")
        
        st.divider()
        st.subheader("Registered Agents")
        
        for i, (name, agent) in enumerate(agents.items(), 1):
            with st.container():
                cols = st.columns([1, 4, 2])
                with cols[0]:
                    st.write(f"**#{i}**")
                with cols[1]:
                    st.write(name)
                with cols[2]:
                    try:
                        params = sum(p.numel() for p in agent.parameters())
                        st.caption(f"{params:,} params")
                    except:
                        st.caption("Random agent")
    
    st.divider()
    
    # Refresh button
    if st.button("🔄 Refresh Agents"):
        st.cache_resource.clear()
        st.rerun()
