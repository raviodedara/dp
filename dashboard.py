import streamlit as st
import pandas as pd
import google.generativeai as genai
import io
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Put this at the very top of your app (after imports)
with st.sidebar:
    st.logo("https://img.icons8.com/cloud/100/data-configuration.png") # Optional Logo
    st.title(" ✈️ Data Pilot AI")

    # 1. The "Bring Your Own Key" Input
    api_key = st.text_input("Enter Gemini API Key", type="password", help="Get a free key at aistudio.google.com")

    # 2. Add a link to your Gumroad
    st.markdown("---")
    st.markdown("Want the Source Code? [**Get it here**](YOUR_GUMROAD_LINK)")

# 3. The Guard Clause (Stops the app if no key)
if not api_key:
    st.info("👈 Please enter your Google Gemini API Key in the sidebar to start the Pilot.")
    st.stop() # This halts the app here. No crashes.

# 4. Configure Gemini with the USER'S key
genai.configure(api_key=api_key)

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Data Pilot AI", page_icon="✈️", layout="wide")

# --- 2. SETUP GEMINI AI ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    ai_available = True
except:
    ai_available = False

@st.cache_data(show_spinner=False)
def get_gemini_response(prompt):
    """Dynamically finds a working model and caches the result."""
    try:
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        # Smart Model Selection
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

# --- 3. CUSTOM CSS (Notebook Style) ---
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
    /* Notebook Cell Style */
    .stChatMessage {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        border: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)

# --- 4. SESSION STATE & HISTORY ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'history' not in st.session_state:
    st.session_state.history = [] # Data History (for Undo)
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [] # Chat History (for Notebook)

def save_data_history():
    """Saves the dataframe state before modification."""
    if st.session_state.df is not None:
        st.session_state.history.append(st.session_state.df.copy())
        if len(st.session_state.history) > 5: # Keep last 5 actions
            st.session_state.history.pop(0)

def undo_action():
    if st.session_state.history:
        st.session_state.df = st.session_state.history.pop()
        # Add a system message to chat history
        st.session_state.chat_history.append({"role": "system", "content": "↺ Undid last data modification."})
        st.success("Action Undone!")
    else:
        st.warning("Nothing to undo.")

# --- 5. SIDEBAR (The Control Panel) ---
with st.sidebar:
    st.markdown("### ✈️ Data Pilot AI")
    uploaded_file = st.file_uploader("📂 Upload Dataset", type=["csv"])
    
    if uploaded_file is not None:
        if 'last_file' not in st.session_state or st.session_state.last_file != uploaded_file.name:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.last_file = uploaded_file.name
            st.session_state.history = [] 
            st.session_state.chat_history = [] # Clear chat on new file
            st.success("New Data Loaded!")
    
    st.markdown("---")
    
    # EXPORT OPTIONS (Realistic Buttons)
    if st.session_state.df is not None:
        st.markdown("**💾 Export Options**")
        csv = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "pilot_data.csv", "text/csv", use_container_width=True)
        st.info("📷 To save charts: Click the camera icon in the top-right of any plot.")
    
    st.markdown("---")
    
    # DATA CONTROLS
    col_undo, col_clear = st.columns(2)
    with col_undo:
        if st.button("↺ Undo", use_container_width=True):
            undo_action()
            st.rerun()
    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# --- 6. MAIN APP ---
st.markdown('<p class="title-text">Data Pilot AI</p>', unsafe_allow_html=True)

if st.session_state.df is not None:
    df = st.session_state.df

    # TAB STRUCTURE
    tab1, tab2 = st.tabs(["📊 Dashboard", "👨🏼‍✈️Data Pilot"])

    # --- TAB 1: SKIMPY DASHBOARD (Unchanged) ---
    with tab1:
        st.markdown("### 🦅 Data Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing", df.isnull().sum().sum())
        c4.metric("Duplicates", df.duplicated().sum())
        st.markdown("---")
        
        num_df = df.select_dtypes(include=['number'])
        if not num_df.empty:
            st.subheader("🔢 Numeric Stats")
            st.dataframe(num_df.describe().T, use_container_width=True)
            
        cat_df = df.select_dtypes(include=['object'])
        if not cat_df.empty:
            st.subheader("🔤 Text Stats")
            st.dataframe(cat_df.describe().T, use_container_width=True)

    # --- TAB 2: Data Pilot (The Major Upgrade) ---
    with tab2:
        st.markdown("### 📓 AI Computational Notebook")
        
        # 1. ANALYST TOOLKIT (Low RPM Actions)
        # These execute LOCAL Python immediately. No API cost.
        with st.expander("🛠️ Analyst Toolkit (Quick Actions)", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            if c1.button("👁️ Show Head"):
                st.session_state.chat_history.append({"role": "user", "content": "Show me the first 5 rows."})
                st.session_state.chat_history.append({"role": "assistant", "code": "st.write(df.head())", "type": "code"})
            
            if c2.button("ℹ️ Data Info"):
                st.session_state.chat_history.append({"role": "user", "content": "Show data types and missing info."})
                # Using a workaround to display df.info() as text
                buffer = io.StringIO()
                df.info(buf=buffer)
                s = buffer.getvalue()
                st.session_state.chat_history.append({"role": "assistant", "code": f"st.text('''{s}''')", "type": "code"})

            if c3.button("📉 Missing Map"):
                st.session_state.chat_history.append({"role": "user", "content": "Check for missing values."})
                st.session_state.chat_history.append({"role": "assistant", "code": "st.write(df.isnull().sum())", "type": "code"})
            
            if c4.button("🧹 Auto-Clean (AI)"):
                # This one uses AI, so we just set the prompt for the chat handler below
                st.session_state.quick_prompt = "Identify missing values. Fill numeric missing values with 0 and text with 'Unknown'. Remove duplicates. Show me the clean head."

        # 2. CHAT HISTORY DISPLAY
        # This creates the "Notebook" feel by showing past interactions
        for msg in st.session_state.chat_history:
            if msg["role"] == "system":
                st.info(msg["content"])
            elif msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            elif msg["role"] == "assistant":
                with st.chat_message("assistant"):
                    if "content" in msg:
                        st.write(msg["content"])
                    if "code" in msg:
                        with st.status("Executed Code", state="complete"):
                            st.code(msg["code"], language="python")
                        # Re-execute code to show output in the notebook stream
                        try:
                            local_scope = {"df": st.session_state.df, "pd": pd, "st": st, "px": px, "plt": plt, "sns": sns}
                            exec(msg["code"], globals(), local_scope)
                        except Exception as e:
                            st.error(f"Error re-rendering: {e}")

        # 3. CHAT INPUT
        # Check if a button triggered a prompt or user typed one
        user_input = st.chat_input("Ask the Pilot (e.g., 'Plot Sales vs Profit')...")
        if 'quick_prompt' in st.session_state:
            user_input = st.session_state.pop('quick_prompt')

        if user_input:
            if not ai_available:
                st.error("⚠️ API Key not found.")
            else:
                # Add User Message to History
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.write(user_input)

                # Generate AI Response
                with st.chat_message("assistant"):
                    with st.spinner("Pilot is coding..."):
                        # Context
                        buffer = io.StringIO()
                        df.info(buf=buffer)
                        info_str = buffer.getvalue()
                        
                        prompt = f"""
                        You are a Python Data Agent using Streamlit. 
                        USER COMMAND: {user_input}
                        DATAFRAME INFO: {info_str}
                        HEADERS: {list(df.columns)}
                        RULES:
                        1. Write valid Python code to manipulate 'df'.
                        2. Use 'plotly.express' as 'px' for plots. Example: fig = px.bar(df, ...); st.plotly_chart(fig)
                        3. Use st.write() to display data/text.
                        4. Output ONLY raw Python code. No markdown.
                        5. If the user asks for a MAP, use 'px.scatter_mapbox'. ALWAYS set mapbox_style="open-street-map" so it works without an API token.
                        """
                        
                        response = get_gemini_response(prompt)
                        
                        if hasattr(response, 'text'):
                            generated_code = response.text.replace("```python", "").replace("```", "").strip()
                            
                            # Execute and Save to History
                            try:
                                save_data_history() # Save Undo state
                                
                                # 1. Show the code block
                                with st.status("Executed Code", state="complete"):
                                    st.code(generated_code, language='python')
                                
                                # 2. Run the code
                                local_scope = {"df": st.session_state.df, "pd": pd, "st": st, "px": px, "plt": plt, "sns": sns}
                                exec(generated_code, globals(), local_scope)
                                
                                # 3. Update State
                                st.session_state.df = local_scope['df']
                                
                                # 4. Append to History (So it stays on refresh)
                                st.session_state.chat_history.append({"role": "assistant", "code": generated_code, "type": "code"})
                                
                            except Exception as e:
                                st.error(f"Execution Error: {e}")
                                st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {e}"})
                        else:
                            st.error(f"AI Error: {response}")

else:
    st.info("👈 Upload a CSV to start the Pilot.")
    