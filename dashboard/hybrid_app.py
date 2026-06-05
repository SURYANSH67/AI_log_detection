import streamlit as st
import pandas as pd
import os
import numpy as np
from streamlit_autorefresh import st_autorefresh
import re
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# PATH CONFIG
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Set page config
st.set_page_config(
    page_title="SOC Hybrid Log Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium SOC Dashboard styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* Global styles */
    .stApp {
        background-color: #0d0e15;
        color: #e2e8f0;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #12131e;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Header decoration */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3b82f6 0%, #10b981 50%, #f43f5e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.05em;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    
    /* Glassmorphism Metric Cards */
    .card-container {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }
    
    .custom-card {
        flex: 1;
        min-width: 250px;
        background: rgba(22, 28, 45, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: left;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-2px);
        border-color: rgba(59, 130, 246, 0.5);
    }
    
    .custom-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
    }
    
    .card-total::before { background: #3b82f6; }
    .card-anomalies::before { background: #ef4444; }
    .card-severity::before { background: #f59e0b; }
    
    .card-title {
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .card-value {
        font-size: 2.25rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.25rem;
    }
    
    .card-desc {
        font-size: 0.8rem;
        color: #64748b;
    }
    
    /* Styled labels */
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.375rem;
    }
    
    .badge-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    
    .badge-medium {
        background-color: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    
    .badge-low {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        background-color: transparent;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border: none;
        color: #94a3b8;
        font-weight: 600;
        font-size: 1rem;
        padding: 0 10px;
        transition: color 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        color: #3b82f6 !important;
        border-bottom: 2px solid #3b82f6 !important;
    }
</style>
""", unsafe_allow_html=True)

# Main Title Section
st.markdown("<h1 class='main-title'>🛡️ SOC Hybrid Log Anomaly Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Real-time multi-source security monitoring & intelligence system (HDFS & Apache logs)</p>", unsafe_allow_html=True)

# Autorefresh simulation (every 5 seconds)
st_autorefresh(interval=5000, key="datarefresh")

# ===============================
# SIDEBAR
# ===============================
st.sidebar.markdown("### ⚙️ Simulation & Engine Settings")
contamination = st.sidebar.slider("Engine Contamination Rate (%)", 1, 20, 5) / 100.0

st.sidebar.info("""
**Explainable AI Engine:**
Uses a hybrid Isolation Forest model trained dynamically. Anomaly decisions are explained by comparing outlier features against baseline normal statistics.
""")

# ===============================
# DATA PROCESSING LAYER
# ===============================
@st.cache_data(ttl=60)
def load_and_extract_features():
    # --- System Logs (HDFS) ---
    hdfs_file = os.path.join(DATA_DIR, "HDFS_2k.log_structured.csv")
    if not os.path.exists(hdfs_file):
        st.error(f"HDFS data missing at {hdfs_file}")
        return pd.DataFrame()
    
    hdfs = pd.read_csv(hdfs_file)
    block_ids = []
    for content in hdfs["Content"]:
        match = re.search(r"blk_-?\d+", str(content))
        block_ids.append(match.group() if match else None)
    
    hdfs["BlockId"] = block_ids
    hdfs = hdfs.dropna(subset=["BlockId"])
    
    system_features = []
    for block_id, group in hdfs.groupby("BlockId"):
        event_count = len(group)
        unique_events = group["EventId"].nunique()
        event_variance = np.std([hash(e) % 1000 for e in group["EventId"]])
        
        if "Time" in group.columns:
            group["Time"] = pd.to_datetime(group["Time"], errors="coerce")
            time_diffs = group["Time"].sort_values().diff().dt.total_seconds()
            avg_time_gap = time_diffs.mean() if not time_diffs.isnull().all() else 0
        else:
            avg_time_gap = 0
            
        system_features.append([
            event_count,
            unique_events,
            event_variance,
            0, # error ratio placeholder
            avg_time_gap,
            block_id
        ])
        
    system_df = pd.DataFrame(
        system_features,
        columns=["f1", "f2", "f3", "f4", "f5", "identifier"]
    )
    system_df["source"] = "System (HDFS)"
    
    # --- Web Logs (Apache) ---
    apache_file = os.path.join(DATA_DIR, "Apache_2k.log_structured.csv")
    if not os.path.exists(apache_file):
        st.error(f"Apache data missing at {apache_file}")
        return system_df
        
    apache = pd.read_csv(apache_file)
    web_features = []
    for template, group in apache.groupby("EventId"):
        request_intensity = len(group)
        error_count = group["Content"].str.contains("error|fail|404|500", case=False).sum()
        error_ratio = error_count / len(group) if len(group) > 0 else 0
        
        web_features.append([
            request_intensity,
            0, # unique events placeholder
            0, # event variance placeholder
            error_ratio,
            0, # avg time gap placeholder
            template
        ])
        
    web_df = pd.DataFrame(
        web_features,
        columns=["f1", "f2", "f3", "f4", "f5", "identifier"]
    )
    web_df["source"] = "Web (Apache)"
    
    # Combined hybrid feature space
    combined_df = pd.concat([system_df, web_df]).reset_index(drop=True)
    return combined_df

combined = load_and_extract_features()

if not combined.empty:
    # Inject slight dynamic noise to simulate real-time sensor updates
    combined["f1"] = combined["f1"] + np.random.randint(0, 2, size=len(combined))
    
    features = combined[["f1", "f2", "f3", "f4", "f5"]]
    
    # Model Training
    model = IsolationForest(contamination=contamination, random_state=42)
    combined["anomaly_raw"] = model.fit_predict(features)
    combined["anomaly"] = combined["anomaly_raw"].map({1: "Normal", -1: "Anomaly"})
    
    # Severity Score
    decision_scores = model.decision_function(features)
    # Scale from 0 to 100 (where lower decision_function value = higher severity anomaly)
    min_score, max_score = decision_scores.min(), decision_scores.max()
    if max_score != min_score:
        severity = (1 - (decision_scores - min_score) / (max_score - min_score)) * 100
    else:
        severity = np.zeros(len(decision_scores))
    combined["severity_score"] = severity.round(2)
    
    def categorize_severity(score):
        if score < 40: return "Low"
        elif score < 70: return "Medium"
        else: return "High"
    combined["severity_level"] = combined["severity_score"].apply(categorize_severity)
    
    # Explainable AI calculations
    normal_data = combined[combined["anomaly"] == "Normal"]
    if not normal_data.empty:
        normal_means = normal_data[["f1", "f2", "f3", "f4", "f5"]].mean()
    else:
        normal_means = pd.Series([1, 1, 1, 1, 1], index=["f1", "f2", "f3", "f4", "f5"])
        
    def generate_explanation(row):
        reasons = []
        if row["f1"] > normal_means["f1"] * 1.5:
            reasons.append("Unusually high event frequency")
        if row["f2"] > normal_means["f2"] * 1.5:
            reasons.append("High diversity of log events")
        if row["f3"] > normal_means["f3"] * 1.5:
            reasons.append("High variance in log behavior")
        if row["f4"] > 0.3:
            reasons.append("High error ratio detected")
        if row["f5"] > normal_means["f5"] * 1.5:
            reasons.append("Abnormal time gap pattern")
        if row["severity_score"] > 70:
            reasons.append("Very far from normal behavior cluster")
        if not reasons:
            reasons.append("Minor statistical deviation")
        return ", ".join(reasons)
        
    combined["explanation"] = combined.apply(generate_explanation, axis=1)
    
    # Filter anomalies
    anomalies_df = combined[combined["anomaly"] == "Anomaly"].copy()
    
    # ===============================
    # RENDER METRIC CARDS
    # ===============================
    total_logs_count = len(combined)
    anomaly_count = len(anomalies_df)
    max_severity = combined["severity_score"].max()
    
    st.markdown(f"""
    <div class="card-container">
        <div class="custom-card card-total">
            <div class="card-title">Total Monitored Units</div>
            <div class="card-value">{total_logs_count}</div>
            <div class="card-desc">Active log tracks evaluated</div>
        </div>
        <div class="custom-card card-anomalies">
            <div class="card-title">Anomalies Detected</div>
            <div class="card-value" style="color: #ef4444;">{anomaly_count}</div>
            <div class="card-desc">Anomalous items needing review</div>
        </div>
        <div class="custom-card card-severity">
            <div class="card-title">Peak Threat Level</div>
            <div class="card-value" style="color: #f59e0b;">{max_severity}%</div>
            <div class="card-desc">Max anomaly severity score</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ===============================
    # TAB INTERFACE
    # ===============================
    tab1, tab2, tab3 = st.tabs([
        "🛡️ Active Threat Feed", 
        "📊 Behavioral Analytics", 
        "⚙️ Engine Diagnostics"
    ])
    
    with tab1:
        st.subheader("🚨 Current Threat Alerts")
        st.write("Below are the detected outliers sorted by threat severity level.")
        
        if not anomalies_df.empty:
            display_anomalies = anomalies_df[[
                "identifier", "source", "severity_score", "severity_level", "explanation"
            ]].sort_values("severity_score", ascending=False)
            
            # Format and present dataframe beautifully
            st.dataframe(
                display_anomalies.style.background_gradient(
                    subset=["severity_score"], cmap="Reds"
                ),
                column_config={
                    "identifier": "Log Unit Identifier",
                    "source": "Log Origin",
                    "severity_score": "Threat Index (%)",
                    "severity_level": "Level",
                    "explanation": "Security Engine Explanation"
                },
                use_container_width=True,
                hide_index=True
            )
            
            # Download Button
            csv_data = anomalies_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Threat Report (CSV)",
                data=csv_data,
                file_name="soc_threat_report.csv",
                mime="text/csv",
            )
        else:
            st.success("✅ Clean bill of health. No active system threats detected.")
            
    with tab2:
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("📈 Hybrid Space Projection")
            st.write("Visualization of combined features f1 (Frequency) and f2 (Unique Events).")
            
            # Matplotlib scatter plot with nice SOC theme
            fig, ax = plt.subplots(figsize=(6, 4.5))
            fig.patch.set_facecolor('#0d0e15')
            ax.set_facecolor('#12131e')
            
            normal = combined[combined["anomaly"] == "Normal"]
            anomaly = combined[combined["anomaly"] == "Anomaly"]
            
            ax.scatter(normal["f1"], normal["f2"], color='#3b82f6', alpha=0.5, label='Normal log path', s=30)
            ax.scatter(anomaly["f1"], anomaly["f2"], color='#f43f5e', alpha=0.9, label='Anomalous signature', s=60, edgecolors='white', linewidths=0.5)
            
            ax.set_xlabel("Feature 1 (Log Event Density)", color='#94a3b8')
            ax.set_ylabel("Feature 2 (Unique Event Diversity)", color='#94a3b8')
            ax.tick_params(colors='#64748b')
            ax.spines['bottom'].color = '#334155'
            ax.spines['top'].visible = False
            ax.spines['right'].visible = False
            ax.spines['left'].color = '#334155'
            ax.grid(True, color='#1e293b', linestyle='--', alpha=0.5)
            
            legend = ax.legend(facecolor='#12131e', edgecolor='#334155')
            for text in legend.get_texts():
                text.set_color('#ffffff')
                
            st.pyplot(fig)
            
        with col_right:
            st.subheader("📊 Feature Variability Dashboard")
            st.write("Displays the statistical variance across hybrid log attributes.")
            
            feature_variance = combined[["f1", "f2", "f3", "f4", "f5"]].var()
            st.bar_chart(feature_variance)
            
            st.subheader("📋 Descriptive Statistics")
            st.dataframe(combined[["f1", "f2", "f3", "f4", "f5"]].describe().T, use_container_width=True)
            
    with tab3:
        st.subheader("🔥 Threat Hotspots (Severity Heatmap)")
        st.write("Average anomaly severity index grouped by log ingestion pipeline.")
        
        severity_map = {"Low": 1, "Medium": 2, "High": 3}
        combined["severity_encoded"] = combined["severity_level"].map(severity_map)
        
        pivot = combined.pivot_table(
            values="severity_encoded",
            index="source",
            aggfunc="mean"
        )
        
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        fig2.patch.set_facecolor('#0d0e15')
        ax2.set_facecolor('#12131e')
        
        sns.heatmap(
            pivot, 
            annot=True, 
            cmap="RdYlGn_r", 
            cbar=False, 
            ax=ax2, 
            annot_kws={"size": 10, "weight": "bold"},
            linewidths=0.5,
            linecolor='#0d0e15'
        )
        ax2.set_title("Average Ingestion Severity Index (1: Low -> 3: High)", color='#ffffff', fontsize=10)
        ax2.tick_params(colors='#94a3b8', labelsize=9)
        
        st.pyplot(fig2)
        
        st.subheader("🧠 Model Baseline Statistics")
        st.write("Below are the baseline normal parameter means computed by the model engine:")
        st.dataframe(normal_means.to_frame(name="Normal Baseline Mean"), use_container_width=True)
else:
    st.error("No data ingested. Please check logs in the data directory.")
