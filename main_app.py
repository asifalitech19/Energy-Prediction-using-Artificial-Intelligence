import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import shap 
import joblib
import os
from datetime import datetime
import time
from supabase import create_client, Client
import base64
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests

# --- CONFIGURATION ---
st.set_page_config(
    page_title="FutureEnergy AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. LOCAL STATE INITIALIZATION ---
def init_session_state():
    if 'users' not in st.session_state:
        st.session_state.users = [
            {"email": "student@gmail.com", "password": "123", "role": "student", "created_at": "2025-07-27"},
            {"email": "teacher@gmail.com", "password": "123", "role": "teacher", "created_at": "2025-07-27"}
        ]

init_session_state()

# --- 2. ASSET LOADING (LOTTIE) ---
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Load Animations
lottie_energy = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_irep8vva.json") 
lottie_ai = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_ofa3xwo7.json") 
lottie_login = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_ucbyrun5.json")
lottie_admin = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_q5pk6p1k.json")

# --- 3. SUPABASE (Optional Fallback) ---
@st.cache_resource
def init_supabase():
    try:
        url = "https://btxhodbiftifoebtfnub.supabase.co"
        key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ0eGhvZGJpZnRpZm9lYnRmbnViIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI1NjczMTIsImV4cCI6MjA2ODE0MzMxMn0.d6Dwd28Ol9v4eGalU1TEYc0ifogZvWsqiiPeKIFtCRA"
        return create_client(url, key)
    except:
        return None

supabase = init_supabase()
ADMIN_EMAIL = "admin@gmail.com"
ADMIN_PASSWORD = "admin@1234"

# --- 4. VIP 3D CSS STYLING ---
def local_css():
    st.markdown("""
    <style>
    /* ANIMATED DEEP SPACE BACKGROUND */
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .stApp {
        background: linear-gradient(-45deg, #0b1120, #1a2a47, #0f1c30, #000000);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        color: white;
    }

    /* 3D FLOATING GLASS CARDS */
    .feature-card, .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        transition: all 0.4s ease;
        margin-bottom: 20px;
    }
    
    .feature-card:hover, .metric-card:hover {
        transform: translateY(-5px) scale(1.01);
        box-shadow: 0 0 20px rgba(76, 175, 80, 0.4); /* Green Glow */
        border: 1px solid rgba(76, 175, 80, 0.5);
    }

    /* SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 21, 46, 0.9);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* NEON BUTTONS */
    .stButton>button {
        background: linear-gradient(90deg, #4CAF50, #2E7D32);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 10px 25px;
        font-weight: bold;
        box-shadow: 0 0 10px rgba(76, 175, 80, 0.3);
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(76, 175, 80, 0.6);
    }

    /* TABS */
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.05);
        border-radius: 10px 10px 0 0;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    
    /* INPUT FIELDS */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.05);
        color: white;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- 5. HELPER FUNCTIONS ---
def show_animated_header(title, icon="⚡", color="#4CAF50"):
    col1, col2 = st.columns([1, 8])
    with col1:
        if lottie_ai:
            st_lottie(lottie_ai, height=80, key=f"head_{title}")
        else:
            st.markdown(f"<h1>{icon}</h1>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <h1 style='color: {color}; text-shadow: 0 0 10px {color}; padding-top: 10px;'>
            {title}
        </h1>
        """, unsafe_allow_html=True)

@st.cache_data
def load_real_energy_data():
    try:
        df = pd.read_csv("new_features.csv")
        return process_data(df)
    except:
        st.warning("⚠️ 'new_features.csv' not found. Using mock data.")
        # Mock data generation for demo purposes if file missing
        dates = pd.date_range(start='1/1/2024', periods=100, freq='H')
        df = pd.DataFrame({
            'datetime': dates,
            'Global_active_power': np.random.uniform(0.5, 5.0, 100),
            'Global_reactive_power': np.random.uniform(0.0, 0.5, 100),
            'Voltage': np.random.uniform(230, 240, 100),
            'Global_intensity': np.random.uniform(1, 20, 100),
            'Sub_metering_1': np.random.uniform(0, 5, 100),
            'Sub_metering_2': np.random.uniform(0, 5, 100),
            'Sub_metering_3': np.random.uniform(0, 5, 100)
        })
        return process_data(df)

def process_data(df):
    df = df.rename(columns={'datetime': 'DateTime', 'Global_active_power': 'Active_Power'})
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['hour'] = df['DateTime'].dt.hour
    df['day_of_week'] = df['DateTime'].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['Apparent_Power'] = np.sqrt(df['Global_reactive_power']**2 + (df['Voltage'] * df['Global_intensity'])**2)
    return df

REAL_DATA = load_real_energy_data()

# --- 6. ML CLASSES (PREDICTOR & TRAINER) ---
class EnergyPredictor:
    def __init__(self, model_path='xgb_energy_model.json'):
        self.model = None
        self.features = ['Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 
                         'Sub_metering_2', 'Sub_metering_3', 'hour', 'day_of_week', 'hour_sin', 'hour_cos', 'Apparent_Power']
        if os.path.exists(model_path):
            try:
                self.model = xgb.XGBRegressor()
                self.model.load_model(model_path)
            except: pass

    def predict(self, input_data):
        if not self.model: return 0.0 # Return 0 if model missing
        input_df = pd.DataFrame([input_data])
        for feature in self.features:
            if feature not in input_df: input_df[feature] = 0
        return self.model.predict(input_df[self.features])[0]

class ModelTrainer:
    def __init__(self):
        self.features = ['Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 
                         'Sub_metering_2', 'Sub_metering_3', 'hour', 'day_of_week', 'hour_sin', 'hour_cos', 'Apparent_Power']
        self.target = 'Active_Power'
    
    def train_model(self, model_type, data, test_size=0.2, random_state=42):
        X = data[self.features]
        y = data[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        if model_type == "XGBoost": model = xgb.XGBRegressor(n_estimators=100, random_state=random_state)
        elif model_type == "Random Forest": model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        else: model = LinearRegression()
        
        start = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start
        y_pred = model.predict(X_test)
        
        metrics = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred),
            "Training Time": f"{training_time:.2f} s"
        }
        return model, metrics

class SHAPVisualizer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.Explainer(model)
    def plot_summary(self, input_data):
        shap_values = self.explainer(input_data)
        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values.values, input_data, feature_names=self.feature_names, show=False)
        st.pyplot(fig)

# --- 7. PAGES (LOGIN, USER, ADMIN) ---

def login_page():
    col1, col2 = st.columns([1, 1.2])
    with col1:
        if lottie_login:
            st_lottie(lottie_login, height=400, key="login_anim")
        else:
            st.image("https://cdn-icons-png.flaticon.com/512/295/295128.png", width=300)
    
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="feature-card">
            <h2 style='text-align: center; color: #4CAF50;'>🔐 Secure Access</h2>
            <p style='text-align: center; color: gray;'>Enter credentials to access the Energy Grid</p>
        </div>
        """, unsafe_allow_html=True)
        
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("🚀 Access Dashboard", use_container_width=True):
            if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
                st.session_state.update({'logged_in': True, 'admin': True, 'email': email})
                st.success("Welcome Admin!")
                st.rerun()
            else:
                user = next((u for u in st.session_state.users if u['email'] == email and u['password'] == password), None)
                if user:
                    st.session_state.update({'logged_in': True, 'admin': False, 'email': email, 'role': user['role']})
                    st.success(f"Welcome {user['role']}!")
                    st.rerun()
                else:
                    st.error("Invalid Credentials")

        if st.button("Create Account", use_container_width=True):
            st.session_state['show_register'] = True

    if st.session_state.get('show_register', False):
        with st.form("reg"):
            st.subheader("New Identity")
            r_email = st.text_input("Email")
            r_pass = st.text_input("Password", type="password")
            if st.form_submit_button("Sign Up"):
                st.session_state.users.append({"email": r_email, "password": r_pass, "role": "student", "created_at": str(datetime.now())})
                st.success("Account Created!")
                st.session_state['show_register'] = False
                st.rerun()

def user_page():
    predictor = EnergyPredictor('xgb_energy_model.json')
    
    # Initialize Session Variables
    defaults = {
        'prediction_history': [], 'input_mode': 'simple', 'electricity_rate': 0.15,
        'energy_goals': [{"name": "Weekly Savings", "target": 10, "current": 0, "unit": "%", "timeframe": "week"}],
        'appliance_data': [{"name": "AC Unit", "power": 1.5, "hours": 8, "quantity": 1}],
        'achievements': {"first_prediction": False, "energy_saver": False, "goal_achiever": False}
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

    # SIDEBAR
    with st.sidebar:
        if lottie_energy: st_lottie(lottie_energy, height=150)
        st.title(f"User: {st.session_state.get('role', 'User').title()}")
        selected = option_menu(
            menu_title=None,
            options=["Prediction", "Goals", "Appliances", "Data Visualization", "Profile"],
            icons=["lightning", "target", "plug", "graph-up", "person"],
            styles={"container": {"background-color": "transparent"}, "nav-link-selected": {"background-color": "#4CAF50"}}
        )
        if st.button("Logout"):
            st.session_state['logged_in'] = False
            st.rerun()

    # 1. PREDICTION
    if selected == "Prediction":
        show_animated_header("Energy Prediction", "🔮")
        
        with st.form("pred_form"):
            col1, col2 = st.columns(2)
            with col1:
                voltage = st.slider("Voltage (V)", 220.0, 250.0, 230.0)
                intensity = st.slider("Current (A)", 1.0, 30.0, 10.0)
                hour = st.slider("Hour", 0, 23, 12)
            with col2:
                reactive = st.number_input("Reactive Power", 0.0, 5.0, 0.1)
                sub_meter = st.number_input("Sub Metering", 0.0, 20.0, 1.0)
                day = st.selectbox("Day", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
            
            if st.form_submit_button("Analyze Consumption"):
                input_data = {
                    'Global_reactive_power': reactive, 'Voltage': voltage, 'Global_intensity': intensity,
                    'Sub_metering_1': sub_meter, 'Sub_metering_2': sub_meter*0.5, 'Sub_metering_3': sub_meter*0.3,
                    'hour': hour, 'day_of_week': ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].index(day),
                    'hour_sin': np.sin(2*np.pi*hour/24), 'hour_cos': np.cos(2*np.pi*hour/24),
                    'Apparent_Power': np.sqrt(reactive**2 + (voltage*intensity)**2)
                }
                pred = predictor.predict(input_data)
                cost = pred * st.session_state.electricity_rate
                
                st.session_state.prediction_history.append({'time': str(datetime.now()), 'pred': pred, 'cost': cost})
                st.session_state.achievements['first_prediction'] = True
                
                # Results Card
                st.markdown(f"""
                <div class="metric-card">
                    <div style="display: flex; justify-content: space-around; text-align: center;">
                        <div><h3>Power Load</h3><h1 style="color: #4CAF50;">{pred:.2f} kW</h1></div>
                        <div><h3>Est. Hourly Cost</h3><h1 style="color: #FF9800;">${cost:.2f}</h1></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # 2. GOALS
    elif selected == "Goals":
        show_animated_header("Sustainability Goals", "🎯", "#FF9800")
        col1, col2 = st.columns(2)
        with col1:
            for goal in st.session_state.energy_goals:
                st.markdown(f'<div class="feature-card"><h4>{goal["name"]}</h4>', unsafe_allow_html=True)
                st.progress(min(goal['current']/goal['target'], 1.0))
                st.caption(f"{goal['current']} / {goal['target']} {goal['unit']}")
                st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            with st.expander("Add New Goal", expanded=True):
                g_name = st.text_input("Goal Name")
                g_target = st.number_input("Target", value=10)
                if st.button("Create Goal"):
                    st.session_state.energy_goals.append({"name": g_name, "target": g_target, "current": 0, "unit": "%"})
                    st.rerun()

    # 3. APPLIANCES
    elif selected == "Appliances":
        show_animated_header("Appliance Manager", "🔌", "#2196F3")
        
        if st.session_state.appliance_data:
            df_apps = pd.DataFrame(st.session_state.appliance_data)
            df_apps['Daily kWh'] = df_apps['power'] * df_apps['hours'] * df_apps['quantity']
            
            fig = px.pie(df_apps, values='Daily kWh', names='name', title='Energy Distribution', hole=0.4, template="plotly_dark")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(df_apps, use_container_width=True)
        
        with st.expander("Add Appliance"):
            c1, c2, c3 = st.columns(3)
            with c1: name = st.text_input("Name")
            with c2: power = st.number_input("kW", 0.1, 5.0, 1.0)
            with c3: hours = st.number_input("Hours", 1, 24, 5)
            if st.button("Add"):
                st.session_state.appliance_data.append({"name": name, "power": power, "hours": hours, "quantity": 1})
                st.rerun()

    # 4. DATA VISUALIZATION
    elif selected == "Data Visualization":
        show_animated_header("Data Analytics", "📊", "#9C27B0")
        
        # 3D Chart
        if not REAL_DATA.empty:
            sample = REAL_DATA.sample(n=min(500, len(REAL_DATA)))
            fig_3d = px.scatter_3d(sample, x='Voltage', y='Global_intensity', z='Active_Power',
                                  color='Active_Power', color_continuous_scale='Viridis',
                                  title="3D Power Analysis (Interactive)")
            fig_3d.update_layout(paper_bgcolor="rgba(0,0,0,0)", scene=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # Time Series
            fig_line = px.line(sample.sort_values('DateTime'), x='DateTime', y='Active_Power', title="Power Trend", template="plotly_dark")
            fig_line.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_line, use_container_width=True)

    # 5. PROFILE
    elif selected == "Profile":
        st.markdown(f'<div class="feature-card"><h2>👤 {st.session_state.email}</h2><p>Role: {st.session_state.get("role")}</p></div>', unsafe_allow_html=True)
        st.markdown("### Achievements")
        cols = st.columns(4)
        badges = [("First Prediction", "✅"), ("Energy Saver", "💡"), ("Goal Achiever", "🎯"), ("Data Explorer", "📊")]
        for i, (name, icon) in enumerate(badges):
            unlocked = list(st.session_state.achievements.values())[i] if i < len(st.session_state.achievements) else False
            opacity = "1" if unlocked else "0.3"
            cols[i].markdown(f'<div class="metric-card" style="opacity: {opacity}; text-align: center;"><h1>{icon}</h1><p>{name}</p></div>', unsafe_allow_html=True)


def admin_page():
    show_animated_header("Admin Dashboard", "⚡", "#4CAF50")
    
    # Initialize users list with default accounts
    if 'users' not in st.session_state:
        st.session_state.users = [
            {"email": "teacher1@cdut.edu.pk", "role": "teacher", "created_at": "2025-07-24", "password": "teacher@123"},
            {"email": "john123@gmail.com",  "role": "student", "created_at": "2025-07-27", "password": "john@123"},
            {"email": "asif.67019@iqra.edu.pk",  "role": "student", "created_at": "2025-07-27", "password": "john@123"},
        ]
    
    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-header">
            <div class="sidebar-title">🔐 Admin Console</div>
            <div style="color: #aaa;">Logged in as: {st.session_state['email']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        selected = option_menu(
            menu_title=None,
            options=["Dashboard", "Model Training", "Data Management", "User Management"],
            icons=["speedometer", "cpu", "database", "people"],
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#8BC34A", "font-size": "18px"}, 
                "nav-link": {
                    "font-size": "16px", 
                    "text-align": "left", 
                    "margin": "5px", 
                    "--hover-color": "#2E7D32"
                },
                "nav-link-selected": {"background-color": "#4CAF50"},
            }
        )
        
        st.markdown("---")
        if st.button("Logout", key="admin_logout"):
            logout()

    # Dashboard Tab
    if selected == "Dashboard":
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("https://cdn-icons-png.flaticon.com/512/2933/2933245.png", width=150)
        with col2:
            st.subheader("System Overview")
        
        # Metrics Cards
        cols = st.columns(3)
        with cols[0]:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Users</h3>
                <h2>{len(st.session_state.users)}</h2>
            </div>
            """, unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Data Records</h3>
                <h2>{len(REAL_DATA):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Active Models</h3>
                <h2>{len(st.session_state.get('models', {}))}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Quick Actions")
        
        action_cols = st.columns(3)
        with action_cols[0]:
            if st.button("🔄 Refresh Data", key="refresh_data"):
                st.cache_data.clear()
                st.rerun()
        with action_cols[1]:
            if st.button("📊 View Raw Data", key="view_raw_data"):
                st.session_state['show_raw_data'] = True
        with action_cols[2]:
            if st.button("📤 Export Data", key="export_data"):
                REAL_DATA.to_csv("energy_data_export.csv", index=False)
                with open("energy_data_export.csv", "rb") as file:
                    st.download_button(
                        label="Download CSV",
                        data=file,
                        file_name="energy_data_export.csv",
                        mime="text/csv"
                    )
        
        if st.session_state.get('show_raw_data', False):
            with st.expander("Raw Data Viewer", expanded=True):
                st.dataframe(REAL_DATA, use_container_width=True)
                if st.button("Close", key="close_raw_data"):
                    st.session_state['show_raw_data'] = False

    # Model Training Tab
    elif selected == "Model Training":
        st.subheader("Model Training Center")
        
        # Model Selection Cards
        col1, col2, col3 = st.columns(3)
        
        model_info = [
            {
                "title": "XGBoost",
                "desc": "Advanced gradient boosting",
                "color": "#4CAF50",
                "icon": "⚡",
                "type": "XGBoost"
            },
            {
                "title": "Random Forest",
                "desc": "Ensemble learning method",
                "color": "#2196F3",
                "icon": "🌲",
                "type": "Random Forest"
            },
            {
                "title": "Linear Regression",
                "desc": "Simple baseline model",
                "color": "#FF9800",
                "icon": "📈",
                "type": "Linear Regression"
            }
        ]
        
        for i, model in enumerate(model_info):
            with [col1, col2, col3][i]:
                container = st.container()
                container.markdown(f"""
                <div style="text-align: center; padding: 20px; border-radius: 10px; 
                            background: #1E1E1E; border-left: 4px solid {model['color']};
                            margin: 10px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
                    <div style="font-size: 40px; margin-bottom: 10px; color: {model['color']};">
                        {model['icon']}
                    </div>
                    <h3 style="color: {model['color']}; margin: 0;">{model['title']}</h3>
                    <p style="color: #aaa; margin: 5px 0 0;">{model['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if container.button("Select Model", key=f"model_{i}"):
                    st.session_state['train_model'] = model['type']
        
        # Model Training Section
        if 'train_model' in st.session_state:
            st.markdown(f"### Training {st.session_state['train_model']} Model")
            
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider("Test Size Ratio", 0.1, 0.5, 0.2, 0.05)
            with col2:
                random_state = st.number_input("Random State", 0, 100, 42)
            
            if st.button("Start Training", key="start_training"):
                with st.spinner(f"Training {st.session_state['train_model']} model..."):
                    trainer = ModelTrainer()
                    model, metrics = trainer.train_model(
                        st.session_state['train_model'], 
                        REAL_DATA,
                        test_size=test_size,
                        random_state=random_state
                    )
                    
                    if 'models' not in st.session_state:
                        st.session_state['models'] = {}
                    st.session_state['models'][st.session_state['train_model']] = {
                        'model': model,
                        'metrics': metrics
                    }
                    
                    st.success("Model trained successfully!")
                    
                    # Display Metrics
                    cols = st.columns(4)
                    metrics_data = [
                        ("MAE", f"{metrics['MAE']:.4f}", "#4CAF50"),
                        ("RMSE", f"{metrics['RMSE']:.4f}", "#2196F3"),
                        ("R2 Score", f"{metrics['R2']:.4f}", "#FF9800"),
                        ("Training Time", metrics['Training Time'], "#9C27B0")
                    ]
                    
                    for i, (name, value, color) in enumerate(metrics_data):
                        with cols[i]:
                            st.markdown(f"""
                            <div class="metric-card" style="border-left: 4px solid {color};">
                                <h3>{name}</h3>
                                <h2>{value}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # SHAP Visualization
                    st.subheader("SHAP Explainability")
                    sample_data = REAL_DATA[trainer.features].sample(5, random_state=random_state)
                    
                    if st.session_state['train_model'] == "Linear Regression":
                        st.info("For Linear Regression, we show coefficient analysis instead of SHAP")
                        coefficients = pd.DataFrame({
                            'Feature': trainer.features,
                            'Coefficient': model.coef_
                        }).sort_values('Coefficient', key=abs, ascending=False)
                        
                        fig = px.bar(coefficients, 
                                    x='Coefficient', 
                                    y='Feature',
                                    orientation='h',
                                    title='Feature Coefficients',
                                    color_discrete_sequence=['#4CAF50'])
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        try:
                            shap_viz = SHAPVisualizer(model, trainer.features)
                            
                            tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Waterfall", "Beeswarm", "Bar"])
                            with tab1:
                                with st.container(height=400):
                                    shap_viz.plot_summary(sample_data)
                            with tab2:
                                with st.container(height=400):
                                    shap_viz.plot_waterfall(sample_data)
                            with tab3:
                                with st.container(height=400):
                                    shap_viz.plot_beeswarm(sample_data)
                            with tab4:
                                with st.container(height=400):
                                    shap_viz.plot_bar(sample_data)
                        except Exception as e:
                            st.error(f"SHAP visualization failed: {str(e)}")
                    
                    # Model Comparison
                    if len(st.session_state.get('models', {})) > 1:
                        st.subheader("Model Comparison")
                        comparison_data = []
                        for model_name, model_data in st.session_state['models'].items():
                            comparison_data.append({
                                'Model': model_name,
                                'MAE': model_data['metrics']['MAE'],
                                'RMSE': model_data['metrics']['RMSE'],
                                'R2': model_data['metrics']['R2'],
                                'Training Time': model_data['metrics']['Training Time']
                            })
                        
                        df_comparison = pd.DataFrame(comparison_data)
                        st.dataframe(df_comparison.style
                                    .highlight_min(subset=['MAE', 'RMSE'], color='lightgreen')
                                    .highlight_max(subset=['R2'], color='lightgreen'),
                                    use_container_width=True)
                        
                        fig = px.bar(df_comparison.melt(id_vars=['Model'], 
                                    value_vars=['MAE', 'RMSE', 'R2']),
                                    x='Model', y='value', color='variable',
                                    barmode='group', title='Model Performance Comparison',
                                    template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)

    # Data Management Tab
    elif selected == "Data Management":
        st.subheader("Data Management")
        
        tab1, tab2 = st.tabs(["Upload Data", "Data Statistics"])
        
        with tab1:
            st.subheader("Upload New Dataset")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    new_data = pd.read_csv(uploaded_file)
                    required_cols = ['datetime', 'Global_active_power']
                    if all(col in new_data.columns for col in required_cols):
                        st.success("Data uploaded successfully!")
                        st.write("Preview:")
                        st.dataframe(new_data.head(3))
                    else:
                        missing = [col for col in required_cols if col not in new_data.columns]
                        st.error(f"Missing required columns: {', '.join(missing)}")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        with tab2:
            st.subheader("Data Statistics")
            st.dataframe(REAL_DATA.describe(), use_container_width=True)
            
            st.subheader("Missing Values Analysis")
            missing = REAL_DATA.isnull().sum().reset_index()
            missing.columns = ['Column', 'Missing Values']
            st.dataframe(missing, use_container_width=True)

    # User Management Tab
    elif selected == "User Management":
        st.subheader("User Management")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Registered Users")
            users_df = pd.DataFrame(st.session_state.users)
            
            # Display editable user table
            edited_df = st.data_editor(
                users_df[['email', 'role', 'created_at']],
                column_config={
                    "email": "Email",
                    "role": st.column_config.SelectboxColumn(
                        "Role",
                        options=["teacher", "technician", "student", "admin"],
                        required=True
                    ),
                    "created_at": "Join Date"
                },
                use_container_width=True,
                height=400,
                key="users_editor"
            )
            
            if st.button("Save Changes", key="save_users"):
                # Update the users in session state
                for i, row in edited_df.iterrows():
                    st.session_state.users[i]['role'] = row['role']
                st.success("User updates saved!")
        
        with col2:
            st.subheader("User Actions")
            
            with st.expander("Add New User", expanded=True):
                with st.form("add_user_form"):
                    new_email = st.text_input("Email Address")
                    new_role = st.selectbox(
                        "Role",
                        ["teacher", "technician", "student", "admin"],
                        index=0
                    )
                    new_password = st.text_input("Password", type="password")
                    
                    if st.form_submit_button("Create User"):
                        if not new_email or not new_password:
                            st.error("Email and password are required!")
                        elif any(u['email'] == new_email for u in st.session_state.users):
                            st.error("User already exists!")
                        else:
                            st.session_state.users.append({
                                "email": new_email,
                                "role": new_role,
                                "password": new_password,
                                "created_at": datetime.now().strftime("%Y-%m-%d")
                            })
                            st.success(f"User {new_email} created successfully!")
                            st.rerun()
            
            with st.expander("Bulk Actions"):
                st.warning("Advanced functionality - use with care")
                
                if st.button("Export User List"):
                    users_df = pd.DataFrame(st.session_state.users)
                    csv = users_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="users_export.csv",
                        mime="text/csv"
                    )
                
                uploaded_users = st.file_uploader("Import Users (CSV)", type="csv")
                if uploaded_users:
                    try:
                        new_users = pd.read_csv(uploaded_users)
                        required_cols = ['email', 'role', 'password']
                        if all(col in new_users.columns for col in required_cols):
                            st.session_state.users.extend(new_users.to_dict('records'))
                            st.success(f"Added {len(new_users)} new users")
                            st.rerun()
                        else:
                            missing = [col for col in required_cols if col not in new_users.columns]
                            st.error(f"Missing columns: {', '.join(missing)}")
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")


# --- MAIN ROUTER ---
if __name__ == "__main__":
    if st.session_state.get('logged_in'):
        if st.session_state.get('admin'):
            admin_page()
        else:
            user_page()
    else:
        login_page()