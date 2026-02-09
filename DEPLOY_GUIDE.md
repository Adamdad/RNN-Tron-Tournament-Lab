# RNN Tron Challenge - Streamlit Cloud Deployment Guide

## Quick Deployment (5 minutes)

### Step 1: Prepare GitHub Repository

```bash
# 1. Create a new GitHub repository (e.g., tron-challenge)
# 2. Copy lab/ folder contents to repository root

# Directory structure should be:
tron-challenge/
├── app_cloud.py           # Main application
├── base_agent.py
├── student_template.py
├── generate_data.py
├── tournament_runner.py
├── submission_manager.py
├── tron_env.py
├── requirements.txt       # Needs to be created
├── packages.txt          # System dependencies (optional)
└── submissions/          # Empty folder
    └── .gitkeep
```

### Step 2: Create requirements.txt

```txt
streamlit>=1.28.0
torch>=2.0.0
numpy>=1.24.0
pygame>=2.5.0
pillow>=10.0.0
```

### Step 3: Create packages.txt (System Dependencies)

```txt
# For pygame
libsdl2-dev
libsdl2-image-dev
libsdl2-mixer-dev
libsdl2-ttf-dev
```

### Step 4: Deploy to Streamlit Cloud

1. Visit [streamlit.io/cloud](https://streamlit.io/cloud)
2. Click "New app"
3. Select your GitHub repository
4. Select `app_cloud.py` as the configuration file
5. Click "Deploy"

**Wait 2-3 minutes for the app to go live!**

---

## Student Usage Workflow

### 1. Develop Model (Local)

```bash
# Student trains locally
python student_template.py --train --epochs 20
```

### 2. Submit to Website

1. Open the Streamlit app link
2. Fill in your name in the sidebar
3. Upload two files:
   - `alice_agent.py` (code)
   - `alice_agent.pth` (weights)
4. Click "Submit & Validate"
5. System automatically validates and displays results

### 3. View Rankings

- Go to "Leaderboard" tab
- Click "Start Tournament" to run the competition
- View real-time rankings

---

## Advanced Configuration

### Persistent Storage (Important!)

Streamlit Cloud files are lost after restart. Two solutions:

#### Option A: Use External Storage (Recommended)

Modify storage path in `app_cloud.py`:

```python
# Use AWS S3
import boto3
s3 = boto3.client('s3')
BUCKET_NAME = 'your-bucket'

# Or use Google Drive
# Or use database (Supabase, Firebase, etc.)
```

#### Option B: Regular Backup to GitHub

Add automatic backup functionality:

```python
# Add to app_cloud.py
import subprocess

def backup_to_github():
    subprocess.run(['git', 'add', 'submissions/'])
    subprocess.run(['git', 'commit', '-m', 'Auto backup'])
    subprocess.run(['git', 'push'])
```

#### Option C: Use Streamlit Persistence (Simplest)

```python
# Use st.session_state and @st.cache_resource
# But files still remain only in memory
```

**Recommendation: For classroom use, Option A is most reliable.**

---

## Customization

### Modify Page Title

In `app_cloud.py`:

```python
st.set_page_config(
    page_title="Your Course Name - RNN Challenge", 
    layout="wide"
)
```

### Modify Scoring Rules

In `tournament_runner.py` modify scoring:

```python
# Current: Win 3 points, Draw 1 point, Loss 0 points
# Can be changed to other rules
```

### Add Parameter Limits

In `validate_submission()` function:

```python
# Current limit: 100K parameters
if total_params > 100_000:
    return False, "Too many parameters", None

# Can add other limits
if inference_time > 0.05:  # 50ms
    return False, "Too slow", None
```

---

## Monitoring and Statistics

### View Submission Logs

In "Submissions" tab you can see:
- Submission time
- Student name
- Model parameters
- Validation status

### Export Results

In Leaderboard page you can add export button:

```python
if st.button("Export Results"):
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

## FAQ

### Q: Files disappear after upload?
A: Streamlit Cloud restarts lose files. Use external storage or regular backup.

### Q: Model loading failed?
A: Check:
1. Is class name `StudentAgent`
2. Is model structure consistent with when it was saved
3. Are files completely uploaded

### Q: Concurrent user limit?
A: Streamlit Cloud free tier supports about 10-20 concurrent users. Upgrade to Team for more.

### Q: How to reset all submissions?
A: Click "Refresh All Agents" in sidebar, or empty `submissions/` folder in GitHub.

---

## Teaching Suggestions

### Course Flow

1. **Week 1**: Release challenge, explain RNN and game rules
2. **Week 2-3**: Students develop locally, submit to website
3. **Week 4**: Run final tournament, announce rankings
4. **Week 5**: Winners share experience

### Prevent Cheating

1. **Code Review**: Regularly check submitted code
2. **Version Control**: Require submission of training logs
3. **Time Limit**: Lock uploads after deadline
4. **Diversity Check**: Detect similar models (using model hash)

### Incentives

- Top 3 on leaderboard get bonus points
- Best Innovation Award (e.g., using Transformer)
- Code Quality Award (best comments and documentation)

---

## Related Links

- Streamlit Cloud: https://streamlit.io/cloud
- Documentation: https://docs.streamlit.io/
- Community: https://discuss.streamlit.io/

---

**Happy Teaching!**
