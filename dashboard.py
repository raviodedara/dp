import streamlit as st
import pandas as pd
import google.generativeai as genai
import io
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Data Pilot", page_icon="✈️", layout="wide")

# --- 2. SESSION STATE SETUP ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'history' not in st.session_state:
    st.session_state.history = [] 
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [] 
if 'user_api_key' not in st.session_state:
    st.session_state.user_api_key = ""

# --- 3. CUSTOM CSS ---
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .title-text {
        font-size: 50px;
        font-weight: 900;
        background: -webkit-linear-gradient(45deg, #1E88E5, #FF4B4B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stChatMessage {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        border: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=50)
    st.title("Data Pilot")
    
    # --- API Key Section ---
    api_input = st.text_input(
        "Enter Gemini API Key", 
        type="password", 
        placeholder="Paste key here...",
        value=st.session_state.user_api_key
    )
    st.markdown("👉 [**Get your Free API Key here**](https://aistudio.google.com/app/apikey)")
    
    if api_input:
        st.session_state.user_api_key = api_input
    
    st.markdown("---")
    
    # File Uploader
    uploaded_file = st.file_uploader("📂 Upload Dataset", type=["csv"])
    if uploaded_file is not None:
        if 'last_file' not in st.session_state or st.session_state.last_file != uploaded_file.name:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.last_file = uploaded_file.name
            st.session_state.history = [] 
            st.session_state.chat_history = []
            st.success("New Data Loaded!")
    
    st.markdown("---")
    
    # Controls
    if st.session_state.df is not None:
        st.markdown("**💾 Export Options**")
        csv = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "pilot_data.csv", "text/csv", use_container_width=True)
        
        col_undo, col_clear = st.columns(2)
        with col_undo:
            if st.button("↺ Undo", use_container_width=True):
                if st.session_state.history:
                    st.session_state.df = st.session_state.history.pop()
                    st.session_state.chat_history.append({"role": "system", "content": "↺ Undid last action"})
                    st.rerun()
        with col_clear:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
            
    st.markdown("---")
    st.markdown("Want the Source Code? [**Get it here**](YOUR_GUMROAD_LINK)")

# --- 5. AI LOGIC (Auto-Detect) ---
def get_gemini_response(prompt):
    try:
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        chosen_model = next((m for m in available_models if 'flash' in m), None)
        if not chosen_model:
            chosen_model = next((m for m in available_models if 'pro' in m), None)
        if not chosen_model and available_models:
            chosen_model = available_models[0]
            
        if chosen_model:
            model = genai.GenerativeModel(chosen_model)
            return model.generate_content(prompt)
        else:
            return "Error: No compatible Gemini models found."
    except Exception as e:
        return f"Connection Error: {str(e)}"

def save_data_history():
    if st.session_state.df is not None:
        st.session_state.history.append(st.session_state.df.copy())
        if len(st.session_state.history) > 5:
            st.session_state.history.pop(0)

# --- 6. MAIN APP ---
st.markdown('<p class="title-text">Data Pilot</p>', unsafe_allow_html=True)

if not st.session_state.user_api_key:
    st.warning("👈 Please enter your Google Gemini API Key in the sidebar to start.")
elif st.session_state.df is None:
    st.info("👈 Upload a CSV file in the sidebar to begin.")
else:
    try:
        genai.configure(api_key=st.session_state.user_api_key)
    except Exception as e:
        st.error(f"API Key Error: {e}")

    df = st.session_state.df
    
    # TABS
    tab1, tab2 = st.tabs(["📊 Dashboard", "👨🏼‍✈️ Analyst Notebook"])

    # --- DASHBOARD ---
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing", df.isnull().sum().sum())
        c4.metric("Duplicates", df.duplicated().sum())
        st.divider()
        st.write("### Data Preview")
        st.dataframe(df.head(), use_container_width=True)

    # --- NOTEBOOK ---
    with tab2:
        st.markdown("### 📓 AI Notebook")
        
        # 1. ANALYST TOOLKIT (RESTORED)
        with st.expander("🛠️ Analyst Toolkit (Quick Actions)", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            if c1.button("👁️ Show Head"):
                st.session_state.chat_history.append({"role": "user", "content": "Show head"})
                st.session_state.chat_history.append({"role": "assistant", "code": "st.write(df.head())", "type": "code"})
            if c2.button("ℹ️ Data Info"):
                st.session_state.chat_history.append({"role": "user", "content": "Show info"})
                buffer = io.StringIO(); df.info(buf=buffer); s = buffer.getvalue()
                st.session_state.chat_history.append({"role": "assistant", "code": f"st.text('''{s}''')", "type": "code"})
            if c3.button("📉 Missing Map"):
                st.session_state.chat_history.append({"role": "user", "content": "Show missing"})
                st.session_state.chat_history.append({"role": "assistant", "code": "st.write(df.isnull().sum())", "type": "code"})
            if c4.button("🧹 Auto-Clean"):
                st.session_state.quick_prompt = "Identify missing values. Fill numeric missing values with 0. Remove duplicates. Show the clean head."

        # 2. CHAT HISTORY
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            elif msg["role"] == "assistant":
                with st.chat_message("assistant"):
                    if "content" in msg: st.write(msg["content"])
                    if "code" in msg:
                        with st.status("Executed Code", state="complete"):
                            st.code(msg["code"], language="python")
                        try:
                            local_scope = {"df": df, "pd": pd, "st": st, "px": px, "plt": plt, "sns": sns}
                            exec(msg["code"], globals(), local_scope)
                        except: pass

        # 3. CHAT INPUT
        user_input = st.chat_input("Ask (e.g., 'Plot Price vs Reviews')...")
        
        # Check for Quick Prompt trigger
        if 'quick_prompt' in st.session_state:
            user_input = st.session_state.pop('quick_prompt')

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    try:
                        buffer = io.StringIO(); df.info(buf=buffer); info_str = buffer.getvalue()
                        
                        prompt = f"""
                        You are an expert Python Data Scientist.
                        Your goal is to write executable Python code to answer the user's question.
                        
                        METADATA:
                        Columns: {list(df.columns)}
                        Data Info: {info_str}
                        
                        USER QUESTION: {user_input}
                        
                        STRICT RULES:
                        1. Use 'plotly.express' as 'px' for visualizations.
                        2. CRITICAL: You MUST display charts using `st.plotly_chart(fig)`.
                        3. CRITICAL: For text/numbers, use `st.write()`.
                        4. MAPS: Use `px.scatter_mapbox`. Set `mapbox_style="open-street-map"` and `zoom=10`.
                        5. Return ONLY Python code. No markdown formatting.
                        """
                        
                        response = get_gemini_response(prompt)
                        
                        if hasattr(response, 'text'):
                            code = response.text.replace("```python", "").replace("```", "").strip()
                            
                            save_data_history()
                            
                            with st.status("Executed Code", state="complete"):
                                st.code(code, language='python')
                            
                            local_scope = {"df": df, "pd": pd, "st": st, "px": px, "plt": plt, "sns": sns}
                            exec(code, globals(), local_scope)
                            
                            st.session_state.df = local_scope['df']
                            st.session_state.chat_history.append({"role": "assistant", "code": code, "type": "code"})
                        else:
                            st.error(f"AI Error: {response}")
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
