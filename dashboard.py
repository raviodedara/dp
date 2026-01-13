import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import io
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn import preprocessing, cluster, decomposition, linear_model
import statsmodels.api as sm
import altair as alt

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Data Pilot Ultra", page_icon="🚀", layout="wide")

# --- 2. SESSION STATE (The Brain's Memory) ---
if 'df' not in st.session_state: st.session_state.df = None
if 'original_df' not in st.session_state: st.session_state.original_df = None
if 'history' not in st.session_state: st.session_state.history = [] 
if 'chat_history' not in st.session_state: st.session_state.chat_history = [] 

# --- 3. CUSTOM CSS (The Professional Look) ---
st.markdown("""
    <style>
    .title-text {
        font-size: 55px; font-weight: 900;
        background: -webkit-linear-gradient(left, #00C9FF, #92FE9D);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .stChatMessage { border-radius: 15px; border: 1px solid #e0e0e0; margin-bottom: 15px; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# --- 4. SIDEBAR (The Control Room) ---
with st.sidebar:
    st.markdown("## 👨🏼‍✈️ Data Pilot Ultra")
    st.caption("Version 2.0 | Advanced Analytics Engine")
    
    # COMMENTED OUT BYOK (Using Secrets as requested)
    # api_input = st.text_input("Enter API Key", type="password")
    
    st.divider()
    uploaded_file = st.file_uploader("📂 Feed the Agent Data", type=["csv", "xlsx"])
    if uploaded_file:
        if 'last_file' not in st.session_state or st.session_state.last_file != uploaded_file.name:
            if uploaded_file.name.endswith('.csv'):
                loaded_df = pd.read_csv(uploaded_file)
            else:
                loaded_df = pd.read_excel(uploaded_file)
            st.session_state.original_df = loaded_df.copy()
            st.session_state.df = loaded_df.copy()
            st.session_state.last_file = uploaded_file.name
            st.session_state.chat_history = []
            st.success("Intelligence Synchronized.")

    if st.session_state.df is not None:
        st.divider()
        st.markdown("### 🛠️ Global Commands")
        if st.button("🔄 Reset Environment"):
            st.session_state.df = st.session_state.original_df.copy()
            st.session_state.history = []
            st.session_state.chat_history.append({"role": "system", "content": "Environment Reset to Original."})
            st.rerun()
        
        csv = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Current Intelligence", data=csv, file_name="datapilot_export.csv", use_container_width=True)

# --- 5. AI EXECUTION LOGIC ---
def get_advanced_ai_response(prompt):
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        # Use Gemini 1.5 Pro for 'World Class' reasoning
        model = genai.GenerativeModel('gemini-1.5-pro')
        return model.generate_content(prompt)
    except Exception as e:
        return f"Intelligence Link Error: {str(e)}"

# --- 6. MAIN INTERFACE ---
st.markdown('<p class="title-text">Data Pilot Ultra</p>', unsafe_allow_html=True)

if "GEMINI_API_KEY" not in st.secrets:
    st.warning("⚠️ Configuration Required: Please add GEMINI_API_KEY to Streamlit Secrets.")
elif st.session_state.df is None:
    st.info("👋 I am your Advanced AI Analyst. Upload a dataset to begin the deep-dive.")
else:
    df = st.session_state.df
    tab1, tab2, tab3 = st.tabs(["📊 Global Dashboard", "🧠 AI Analyst Notebook", "📜 Knowledge Audit"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Data Points", df.size)
        c2.metric("Features", df.shape[1])
        c3.metric("Completeness", f"{(1 - df.isnull().mean().mean())*100:.1f}%")
        c4.metric("Skewness", f"{df.skew(numeric_only=True).mean():.2f}")
        st.write("### Active Intelligence Layer")
        st.dataframe(df, use_container_width=True)

    with tab2:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                if "content" in msg: st.write(msg["content"])
                if "code" in msg: st.code(msg["code"], language="python")

        user_input = st.chat_input("Command the Pilot (e.g., 'Run a clustering analysis on buyers' or 'Predict sales next month')...")
        
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.rerun()

        # Processing the latest input
        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
            latest_query = st.session_state.chat_history[-1]["content"]
            
            with st.chat_message("assistant"):
                with st.spinner("Processing World-Class Analysis..."):
                    context = f"Columns: {list(df.columns)}\nTypes: {df.dtypes.to_dict()}\nStats: {df.describe().to_dict()}"
                    
                    system_prompt = f"""
                    ROLE: Principal Data Scientist & Scholar.
                    CAPABILITY: All Python Data Science libraries (Pandas, Scikit-Learn, Statsmodels, Plotly, Seaborn).
                    
                    ENVIRONMENT:
                    - 'df': Active Data (modifiable).
                    - 'original_df': Raw Data (immutable).
                    - 'st': Streamlit (for UI).
                    - 'px', 'go': Plotly (for interactive visuals).
                    
                    TASK: {latest_query}
                    
                    RULES:
                    1. Use SCHOLARLY METHODS (Clustering, Regression, ANOVA, Time-Series) where applicable.
                    2. VISUALS: Always use `st.plotly_chart()` for interactive charts.
                    3. CLEANING: If asked to clean/edit, assign back to 'df'.
                    4. Output ONLY valid Python code inside triple backticks.
                    """
                    
                    response = get_advanced_ai_response(system_prompt)
                    if hasattr(response, 'text'):
                        code = response.text.split("```python")[1].split("```")[0].strip()
                        
                        try:
                            # SAVE HISTORY BEFORE EXECUTION
                            st.session_state.history.append(df.copy())
                            
                            # EXECUTION SCOPE
                            scope = {"df": df, "original_df": st.session_state.original_df, "pd": pd, "np": np, 
                                     "st": st, "px": px, "go": go, "plt": plt, "sns": sns, "alt": alt, "sm": sm}
                            exec(code, scope)
                            
                            # UPDATE STATE
                            st.session_state.df = scope["df"]
                            st.session_state.chat_history.append({"role": "assistant", "content": "Analysis Complete.", "code": code})
                            st.rerun()
                        except Exception as e:
                            st.error(f"Logic Sync Error: {e}")
                    else:
                        st.error("Intelligence Link Failed.")

    with tab3:
        st.write("### Technical Audit & Provenance")
        st.write("This environment is powered by **Data Pilot Ultra**, utilizing Global Data Science standards.")
        st.json(df.dtypes.astype(str).to_dict())
