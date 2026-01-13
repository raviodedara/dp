import streamlit as st
import pandas as pd
import numpy as np
from google import genai # 2026 Stable Library
import io
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn import preprocessing, cluster, decomposition, linear_model
import statsmodels.api as sm
import altair as alt

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Data Pilot Ultra", page_icon="🚀", layout="wide")

# --- 2. SESSION STATE ---
if 'df' not in st.session_state: st.session_state.df = None
if 'original_df' not in st.session_state: st.session_state.original_df = None
if 'history' not in st.session_state: st.session_state.history = [] 
if 'chat_history' not in st.session_state: st.session_state.chat_history = [] 

# --- 3. CUSTOM CSS ---
st.markdown("""
    <style>
    .title-text {
        font-size: 55px; font-weight: 900;
        background: -webkit-linear-gradient(left, #00C9FF, #92FE9D);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .stChatMessage { border-radius: 15px; border: 1px solid #e0e0e0; margin-bottom: 15px; }
    .error-log { background-color: #ffeeee; border: 1px solid #ff0000; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 0.8rem; }
    </style>
""", unsafe_allow_html=True)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("## 👨🏼‍✈️ Data Pilot Ultra")
    st.caption("Version 4.0 | 2026 Intelligence Core")
    
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        api_key = st.text_input("Enter Gemini API Key", type="password")
    
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
        st.markdown("**💾 Export & Control**")
        
        col_undo, col_reset = st.columns(2)
        with col_undo:
            if st.button("↺ Undo", width="stretch"):
                if st.session_state.history:
                    st.session_state.df = st.session_state.history.pop()
                    st.rerun()
        with col_reset:
            if st.button("🔄 Reset", width="stretch"):
                st.session_state.df = st.session_state.original_df.copy()
                st.session_state.history = []
                st.rerun()
        
        csv = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export CSV", data=csv, file_name="datapilot_export.csv", width="stretch")

# --- 5. AI EXECUTION LOGIC ---
def get_advanced_ai_response(prompt, key):
    try:
        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model='gemini-2.0-flash', 
            contents=prompt
        )
        return response, None
    except Exception as e:
        return None, str(e)

# --- 6. MAIN INTERFACE ---
st.markdown('<p class="title-text">Data Pilot Ultra</p>', unsafe_allow_html=True)

if not api_key:
    st.warning("👈 Provide a Gemini API Key to activate.")
elif st.session_state.df is None:
    st.info("👋 Upload a dataset to begin.")
else:
    df = st.session_state.df
    tab1, tab2 = st.tabs(["📊 Dashboard", "🧠 Analyst Notebook"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", df.shape[0])
        c2.metric("Features", df.shape[1])
        c3.metric("Completeness", f"{(1 - df.isnull().mean().mean())*100:.1f}%")
        c4.metric("Skewness", f"{df.skew(numeric_only=True).mean():.2f}")
        st.write("### Active Intelligence Layer")
        st.dataframe(df.head(10), width="stretch")

    with tab2:
        # ANALYST TOOLKIT
        with st.expander("🛠️ Analyst Toolkit", expanded=False):
            t1, t2, t3, t4 = st.columns(4)
            if t1.button("👁️ Preview", width="stretch"):
                st.session_state.chat_history.append({"role": "user", "content": "Show data preview"})
                st.session_state.chat_history.append({"role": "assistant", "code": "st.write(df.head())"})
            if t2.button("📉 Missing", width="stretch"):
                st.session_state.chat_history.append({"role": "user", "content": "Analyze missing values"})
                st.session_state.chat_history.append({"role": "assistant", "code": "st.write(df.isnull().sum())"})
            if t3.button("🧹 Clean", width="stretch"):
                st.session_state.quick_prompt = "Clean data: fix types and handle missing values. Assign to 'df'."
            if t4.button("📊 Stats", width="stretch"):
                st.session_state.chat_history.append({"role": "user", "content": "Show descriptive statistics"})
                st.session_state.chat_history.append({"role": "assistant", "code": "st.write(df.describe())"})

        # CHAT DISPLAY
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                if "content" in msg: st.write(msg["content"])
                if "code" in msg:
                    with st.status("Executed Code", state="complete"):
                        st.code(msg["code"], language="python")
                    try:
                        scope = {"df": df, "original_df": st.session_state.original_df, "pd": pd, "np": np, "st": st, "px": px, "plt": plt, "sns": sns, "go": go, "sm": sm, "alt": alt}
                        exec(msg["code"], scope)
                    except: pass

        user_input = st.chat_input("Command the Pilot...")
        if 'quick_prompt' in st.session_state:
            user_input = st.session_state.pop('quick_prompt')

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    prompt = f"""
                    ROLE: Principal Data Scientist.
                    ENVIRONMENT: 'df' (active), 'original_df' (raw), 'st', 'px', 'go', 'sm', 'alt', 'sns'.
                    METADATA: {list(df.columns)}
                    TASK: {user_input}
                    RULES: Return ONLY raw Python code inside triple backticks. Use st.plotly_chart() for visuals. Update 'df' for edits.
                    """
                    response, error = get_advanced_ai_response(prompt, api_key)
                    
                    if error:
                        st.error("Intelligence Link Failed.")
                        with st.expander("🛠️ Error Logger (System Details)"):
                            st.markdown(f'<div class="error-log">{error}</div>', unsafe_allow_html=True)
                    elif hasattr(response, 'text'):
                        try:
                            code = response.text.split("```python")[-1].split("```")[0].strip()
                            st.session_state.history.append(df.copy())
                            
                            scope = {"df": df, "original_df": st.session_state.original_df, "pd": pd, "np": np, "st": st, "px": px, "plt": plt, "sns": sns, "go": go, "sm": sm, "alt": alt}
                            exec(code, scope)
                            
                            st.session_state.df = scope["df"]
                            st.session_state.chat_history.append({"role": "assistant", "code": code})
                            st.rerun()
                        except Exception as e:
                            st.error("Python Execution Error.")
                            with st.expander("🛠️ Error Logger (Code Details)"):
                                st.code(code, language="python")
                                st.markdown(f'<div class="error-log">{traceback.format_exc()}</div>', unsafe_allow_html=True)
