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
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# --- 4. SIDEBAR (Restored Features) ---
with st.sidebar:
    st.markdown("## 👨🏼‍✈️ Data Pilot Ultra")
    st.caption("Version 2.0 | Advanced Analytics Engine")
    
    # API KEY FAILSAFE: If Secrets missing, show input box
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        api_key = st.text_input("Enter Gemini API Key (Secrets not found)", type="password")
        if api_key:
            st.info("Using manually entered key.")
    
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
        
        # RESTORED UNDO BUTTON
        col_undo, col_reset = st.columns(2)
        with col_undo:
            if st.button("↺ Undo", use_container_width=True):
                if st.session_state.history:
                    st.session_state.df = st.session_state.history.pop()
                    st.rerun()
        with col_reset:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.df = st.session_state.original_df.copy()
                st.session_state.history = []
                st.rerun()
        
        csv = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export CSV", data=csv, file_name="datapilot_export.csv", use_container_width=True)

# --- 5. AI EXECUTION LOGIC ---
def get_advanced_ai_response(prompt, key):
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        return model.generate_content(prompt)
    except Exception as e:
        return f"Intelligence Link Error: {str(e)}"

# --- 6. MAIN INTERFACE ---
st.markdown('<p class="title-text">Data Pilot Ultra</p>', unsafe_allow_html=True)

if not api_key:
    st.warning("👈 Please provide a Gemini API Key to activate the Pilot.")
elif st.session_state.df is None:
    st.info("👋 Upload a dataset in the sidebar to begin the deep-dive.")
else:
    df = st.session_state.df
    tab1, tab2, tab3 = st.tabs(["📊 Global Dashboard", "🧠 AI Analyst Notebook", "📜 Knowledge Audit"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", df.shape[0])
        c2.metric("Features", df.shape[1])
        c3.metric("Completeness", f"{(1 - df.isnull().mean().mean())*100:.1f}%")
        c4.metric("Skewness", f"{df.skew(numeric_only=True).mean():.2f}")
        st.write("### Active Intelligence Layer")
        st.dataframe(df.head(10), use_container_width=True)

    with tab2:
        # RESTORED ANALYST TOOLKIT (Quick Actions)
        with st.expander("🛠️ Analyst Toolkit (Quick Actions)", expanded=False):
            t1, t2, t3, t4 = st.columns(4)
            if t1.button("👁️ Preview Data"):
                st.session_state.chat_history.append({"role": "user", "content": "Show data preview"})
                st.session_state.chat_history.append({"role": "assistant", "code": "st.write(df.head())"})
            if t2.button("📉 Missing Map"):
                st.session_state.chat_history.append({"role": "user", "content": "Analyze missing values"})
                st.session_state.chat_history.append({"role": "assistant", "code": "st.write(df.isnull().sum())"})
            if t3.button("🧹 Auto-Clean"):
                st.session_state.quick_prompt = "Perform scholarly data cleaning: handle missing values, remove duplicates, and fix data types. Assign to 'df'."
            if t4.button("📊 Correlation"):
                st.session_state.chat_history.append({"role": "user", "content": "Show correlation heatmap"})
                st.session_state.chat_history.append({"role": "assistant", "code": "fig = px.imshow(df.corr(numeric_only=True)); st.plotly_chart(fig)"})

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                if "content" in msg: st.write(msg["content"])
                if "code" in msg:
                    with st.status("Analysis Executed", state="complete"):
                        st.code(msg["code"], language="python")
                    # Immediate re-execution for UI consistency
                    try:
                        scope = {"df": df, "original_df": st.session_state.original_df, "pd": pd, "np": np, "st": st, "px": px, "plt": plt, "sns": sns}
                        exec(msg["code"], scope)
                    except: pass

        user_input = st.chat_input("Command the Pilot...")
        if 'quick_prompt' in st.session_state:
            user_input = st.session_state.pop('quick_prompt')

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)
            
            with st.chat_message("assistant"):
                with st.spinner("Executing Scholar-Level Analysis..."):
                    context = f"Columns: {list(df.columns)}\nTypes: {df.dtypes.to_dict()}"
                    prompt = f"""
                    ROLE: Principal Data Scientist.
                    ENVIRONMENT: 'df' (active), 'original_df' (raw), 'px' (Plotly Express).
                    METADATA: {context}
                    TASK: {user_input}
                    RULES: Return ONLY raw Python code. Use `st.plotly_chart()` for graphs. Update 'df' if requested to clean/change.
                    """
                    response = get_advanced_ai_response(prompt, api_key)
                    
                    if hasattr(response, 'text'):
                        code = response.text.replace("```python", "").replace("```", "").strip()
                        st.session_state.history.append(df.copy())
                        
                        try:
                            scope = {"df": df, "original_df": st.session_state.original_df, "pd": pd, "np": np, "st": st, "px": px, "plt": plt, "sns": sns, "go": go, "sm": sm}
                            exec(code, scope)
                            st.session_state.df = scope["df"]
                            st.session_state.chat_history.append({"role": "assistant", "code": code})
                            st.rerun()
                        except Exception as e:
                            st.error(f"Execution Error: {e}")
                    else:
                        st.error("Intelligence Link Failed. Check API Key.")

    with tab3:
        st.write("### Technical Knowledge Audit")
        st.json(df.dtypes.astype(str).to_dict())
