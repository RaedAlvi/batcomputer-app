import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import base64

# Page configuration and styling
st.set_page_config(
    page_title="Batcomputer | Stock Analysis",
    page_icon="🦇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Batman theme
css = """
<style>
    /* Main background */
    .stApp {
        background-color: #0D1117;
        background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), 
                          url('batComputer.png');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #FFC107 !important;
        font-family: 'Segoe UI', sans-serif;
    }
    
    h1 {
        text-shadow: 0 0 10px #FFC107;
        letter-spacing: 2px;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1lcbmhc {
        background-color: rgba(20, 20, 35, 0.8);
    }
    
    /* Custom button styling */
    .stButton > button {
        background-color: #FFC107;
        color: #121212;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #FFD700;
        transform: translateY(-2px);
        box-shadow: 0 0 15px #FFC107;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        border: 1px dashed #FFC107;
        border-radius: 10px;
        background-color: rgba(20, 20, 35, 0.5);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(20, 20, 35, 0.7);
        border-radius: 5px;
        border-left: 5px solid #FFC107;
    }
    
    /* Metric styling */
    [data-testid="stMetric"] {
        background-color: rgba(20, 20, 35, 0.7);
        border-radius: 10px;
        padding: 10px !important;
        border-left: 5px solid #FFC107;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    .batman-section h2 {
    color: #FFC107;
    font-size: 2rem;
    font-weight: bold;
    letter-spacing: 1.5px;
    margin: 0;
    text-shadow: 0 0 8px #FFC107;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

def main():
    def show_batman_intro():
    # Batman intro animation
     st.markdown("""
    <style>
    @keyframes batSignal {
        0% { opacity: 0; transform: scale(0.5); }
        50% { opacity: 1; transform: scale(1.2); }
        100% { opacity: 0; transform: scale(0.5); }
    }
    .bat-signal {
        animation: batSignal 3s ease-in-out;
        text-align: center;
    }
    </style>
    
    <div class="bat-signal">
        <img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExYXRobHM1bzlxMDkyeWw0ZDBtZmpoNXEwNGl6dDc2cmZzeXQxb2s2YiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/Ykjhl76yd1YTsMer73/giphy.gif" width="200">
        <h2 style="color: #FFC107;">Initializing Batcomputer...</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Simulate loading
    import time
    time.sleep(2)
    st.empty()  # Clear the animation

# Call this at the beginning of your main function
    show_batman_intro()

    # ─── Page Setup ───────────────────────────────────────────────────
    st.title("Welcome to the Batcomputer🦇")
    st.markdown("""
    This Supercomputer allows you to analyze and predict stock prices with the power of machine learning, 
    just like Gotham's very own tech mastermind - **Batman**. 
    """)
    st.image(r"C:\Users\raeda\Downloads\Python\lockIn\pff\batman.gif", width=300)

    # ─── Upload Dataset ─────────────────────────────────────────────────
    uploaded_file = st.file_uploader("Upload CSV Dataset:", type=["csv"])
    if not uploaded_file:
        st.write("📁 **How to Use:** Upload a CSV file and select features in the sidebar.")
        return

    try:
        df = pd.read_csv(uploaded_file)
        st.write("### 1. Dataset Preview")
        st.dataframe(df.head(), height=250)

        # ─── 1. Sidebar Feature Selection ────────────────────────────────
        st.sidebar.header("⚙️ Select Features")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        target = st.sidebar.selectbox("Select Target Feature (DV):", numeric_cols)
        available_feats = [c for c in numeric_cols if c != target]
        features = st.sidebar.multiselect(
            "Select Input Features (IVs):",
            options=available_feats,
            default=available_feats[:3]
        )
        if not features:
            st.sidebar.error("Please select at least one feature")
            return

        st.session_state.data = df
        st.session_state.features = features
        st.session_state.target = target
        st.session_state.steps = {'loaded': True, 'processed': True}
        st.success("Features and target confirmed!")

        # ─── 2. Data Visualization Inputs ────────────────────────────────
        st.sidebar.header("⚙️ Select Features for Visualization")
        non_numeric_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        x_axis = st.sidebar.selectbox("Select X-axis Column:", numeric_cols, index=0)
        y_axis = st.sidebar.selectbox("Select Y-axis Column:", numeric_cols, index=1)
        color_column = st.sidebar.selectbox("Select Column for Color Grouping:", non_numeric_cols or ["None"])

        # ─── 2. Data Visualization ───────────────────────────────────────
        with st.expander("2. Data Visualization"):
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                st.line_chart(df.set_index('Date')[y_axis])
            else:
                st.scatter_chart(data=df, x=x_axis, y=y_axis,
                                 color=(color_column if non_numeric_cols else None))

        # ─── 3. Data Analysis ────────────────────────────────────────────
        if st.session_state.steps.get('processed'):
            st.header("3. Data Analysis")
            df = st.session_state.data
            features = st.session_state.features
            target = st.session_state.target

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Feature‑Target Relationships")
                feat = st.selectbox("Select feature to plot:", features)

                # Base scatter
                fig = px.scatter(df, x=feat, y=target, height=400,
                                 labels={feat: feat, target: target})

                # Overlay sklearn trendline
                X = df[[feat]].dropna().values
                y_vals = df.loc[df[feat].notna(), target].values
                lr = LinearRegression().fit(X, y_vals)
                y_pred = lr.predict(X)
                fig.add_trace(
                    go.Scatter(x=X.flatten(), y=y_pred, mode='lines', name='Trendline')
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Correlation Matrix")
                corr = df[features + [target]].corr()
                fig2 = px.imshow(
                    corr,
                    text_auto=".2f",
                   color_continuous_scale=['#0D1117', '#16213E', '#1A237E', '#FFC107'],
                    aspect="auto",
                    labels=dict(x="Feature", y="Feature", color="Correlation")
                )
                st.plotly_chart(fig2, use_container_width=True)

            if st.button("🚀 Proceed to Model Training"):
                st.session_state.steps['ready_for_model'] = True

        # ─── 4. Model Training & Evaluation ──────────────────────────────
        if st.session_state.steps.get('ready_for_model'):
            st.markdown("""
<div class="batman-section">
    <h2>4. Model Training & Evaluation</h2>
</div>
""", unsafe_allow_html=True)

            with st.spinner("⚡ The Batcomputer is processing 🦇..."):

             X = df[features]
             y = df[target]
             X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
             )

             scaler = StandardScaler()
             X_train_scaled = scaler.fit_transform(X_train)
             X_test_scaled = scaler.transform(X_test)

             model = LinearRegression().fit(X_train_scaled, y_train)
             y_pred = model.predict(X_test_scaled)

             st.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
             st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

             fig3 = px.scatter(
                 x=np.arange(len(y_test)),
                 y=y_test,
                 labels={"x": "Sample", "y": target},
                 title="Actual vs Predicted"
             )
             # Add predicted values
             fig3.add_scatter(
                  x=np.arange(len(y_pred)),
                  y=y_pred,
                  mode="markers",
                 name="Predicted",
              marker=dict(color='#FFC107', size=8)  # Batman yellow
) 

# Update layout with Batman theme
             fig3.update_layout(
                 template="plotly_dark",
                  plot_bgcolor='#020205',          # Dark background
                  paper_bgcolor='#020205',         # Dark paper background
                  font=dict(color='#C9BFB1'),      # Light tan text
                 title_font_color='#FFC107'       # Batman yellow title
)

# Update axes with themed grid lines
            fig3.update_xaxes(
                 gridcolor='#595E56',             # Grayish grid
                 zerolinecolor='#595E56'          # Grayish zero line
)
            fig3.update_yaxes(
                 gridcolor='#595E56',             # Grayish grid
                zerolinecolor='#595E56'          # Grayish zero line
)
        st.plotly_chart(fig3, use_container_width=True)

        # ─── 5. Download Predictions ──────────────────────────────────────
        if st.session_state.steps.get('ready_for_model'):
            st.header("5. Download Predictions")
            preds_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
            csv = preds_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
            st.markdown("""
    <div style='
        border: 3px solid #FFC107;
        border-radius: 15px;
        padding: 20px;
        margin: 30px 0;
        background-color: #1A1A2E;
        text-align: center;
        box-shadow: 0 0 25px #FFC107;
    '>
        <h2 style='
            color: #FFC107;
            font-family: Arial Black;
            text-shadow: 0 0 10px #FFC107;
            letter-spacing: 2px;
        '>🦇 THE BAT PROTOCOL HAS BEEN COMPLETED 🦇</h2>
        <img src="https://media0.giphy.com/media/uHo3T7P0YX7Ec/giphy.gif" 
             style='
                 border-radius: 10px;
                 margin: 20px auto;
                 max-width: 600px;
             '>
    </div>
    """, unsafe_allow_html=True)


    except Exception as e:
        st.error(f"Error loading file: {e}")

    # Footer
    st.markdown("""
    <div class='feature-selector'>
    📁 **How does the Bat-Computer Function:**
    1. Upload any CSV or Excel file with numeric data  
    2. Select target variable (what you want to predict)  
    3. Choose features (variables used for prediction)  
    4. Activate the Bat Protocol
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
