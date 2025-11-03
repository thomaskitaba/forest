import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Page configuration - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Forest Growth Monitor",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

try:
    from ml_utils import ForestGrowthPredictor
    st.success("✅ Successfully imported ForestGrowthPredictor!")
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    # Define the class inline if import fails
    exec('''
import numpy as np
import pandas as pd
import warnings
import streamlit as st

warnings.filterwarnings('ignore')

class ForestGrowthPredictor:
    def __init__(self):
        self.is_trained = False
        self.trend_params = None
        
    def train(self, data, **kwargs):
        try:
            df = pd.DataFrame(data)
            ndvi = df["ndvi"].values
            years = df["year"].values
            
            # Simple linear trend
            n = len(ndvi)
            x = np.arange(n)
            y = ndvi
            
            slope = (n * np.sum(x*y) - np.sum(x)*np.sum(y)) / (n * np.sum(x*x) - np.sum(x)**2)
            intercept = np.mean(y) - slope * np.mean(x)
            
            self.trend_params = {"slope": slope, "intercept": intercept}
            self.is_trained = True
            
            # Calculate predictions for metrics
            pred = intercept + slope * x
            r2 = 1 - np.sum((y - pred)**2) / np.sum((y - np.mean(y))**2)
            
            return {
                "r2": max(0, r2),
                "mae": np.mean(np.abs(y - pred)),
                "train_size": len(ndvi),
                "test_size": 0,
                "model_type": "Linear Trend"
            }
        except Exception as e:
            st.error(f"Training failed: {e}")
            return None
            
    def predict_future(self, data, years_ahead=3):
        if not self.is_trained:
            return None
            
        df = pd.DataFrame(data)
        last_year = df["year"].iloc[-1]
        
        predictions = []
        for i in range(1, years_ahead + 1):
            pred = self.trend_params["intercept"] + self.trend_params["slope"] * (len(df) + i - 1)
            predictions.append(max(0.1, min(0.9, pred)))
            
        return np.array(predictions)
        
    def get_model_info(self):
        return {
            "architecture": "Linear Trend",
            "trained": self.is_trained,
            "method": "Statistical Analysis"
        }
''')
    from ml_utils import ForestGrowthPredictor
    st.success("✅ Created ForestGrowthPredictor inline!")

# Mock data and functions
def initialize_earth_engine():
    return True

def get_forest_data(geometry):
    return [
        {'year': 2018, 'ndvi': 0.65, 'forest_cover': 56},
        {'year': 2019, 'ndvi': 0.68, 'forest_cover': 58},
        {'year': 2020, 'ndvi': 0.71, 'forest_cover': 62},
        {'year': 2021, 'ndvi': 0.69, 'forest_cover': 60},
        {'year': 2022, 'ndvi': 0.73, 'forest_cover': 65},
        {'year': 2023, 'ndvi': 0.75, 'forest_cover': 68},
        {'year': 2024, 'ndvi': 0.77, 'forest_cover': 70}
    ]

def create_aoi_geometry(min_lon, max_lon, min_lat, max_lat):
    return {"type": "Polygon", "coordinates": [[
        [min_lon, min_lat], [max_lon, min_lat],
        [max_lon, max_lat], [min_lon, max_lat],
        [min_lon, min_lat]
    ]]}

# Mock AOIS
AOIS = {
    'Bale Mountains': {'min_lon': 39.0, 'max_lon': 40.0, 'min_lat': 6.0, 'max_lat': 7.5},
    'Simien Mountains': {'min_lon': 38.0, 'max_lon': 39.0, 'min_lat': 13.0, 'max_lat': 14.0}
}

def main():
    st.title("🌳 Forest Growth Monitoring System")
    st.markdown("### Lightweight Statistical Analysis")
    
    # Initialize session state
    if 'ml_model' not in st.session_state:
        st.session_state.ml_model = ForestGrowthPredictor()
    
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", ["🏠 Dashboard", "📊 Data Analysis", "🤖 ML Predictions"])
    
    # AOI Selection
    st.sidebar.subheader("📍 Area Selection")
    aoi_names = ["Custom"] + list(AOIS.keys())
    selected_aoi = st.sidebar.selectbox("Select Area", aoi_names)
    
    # Demo data
    demo_data = get_forest_data(None)
    
    if app_mode == "🏠 Dashboard":
        show_dashboard(demo_data)
    elif app_mode == "📊 Data Analysis":
        show_data_analysis(demo_data)
    elif app_mode == "🤖 ML Predictions":
        show_ml_predictions(demo_data)

def show_dashboard(data):
    st.header("🌿 Forest Growth Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📍 Study Area")
        st.info("Bale Mountains Region, Ethiopia")
        st.metric("Area Size", "1,200 km²")
        st.metric("Planting Started", "2018")
    
    with col2:
        st.subheader("📈 Growth Metrics")
        df = pd.DataFrame(data)
        current_ndvi = df['ndvi'].iloc[-1]
        ndvi_change = current_ndvi - df['ndvi'].iloc[0]
        
        st.metric("Current NDVI", f"{current_ndvi:.3f}", f"{ndvi_change:+.3f}")
        st.metric("Forest Cover", f"{df['forest_cover'].iloc[-1]}%", "+12%")
        st.metric("Data Points", len(df))
    
    with col3:
        st.subheader("🎯 Analysis Status")
        if st.session_state.trained:
            st.success("✅ Model Ready")
            model_info = st.session_state.ml_model.get_model_info()
            st.info(f"Method: {model_info['method']}")
        else:
            st.warning("🤖 Train Model")
        
        st.metric("Analysis Type", "Statistical")
        st.metric("Python Version", "3.13")
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 View Data", use_container_width=True):
            st.session_state.current_page = "data"
            st.rerun()
    
    with col2:
        if st.button("🤖 Train Model", use_container_width=True):
            results = st.session_state.ml_model.train(demo_data)
            if results:
                st.session_state.trained = True
                st.success("Model trained successfully!")
    
    with col3:
        if st.button("🔮 Predict", use_container_width=True):
            if st.session_state.trained:
                predictions = st.session_state.ml_model.predict_future(demo_data, 3)
                if predictions is not None:
                    st.success("Predictions generated!")
            else:
                st.warning("Please train the model first")

def show_data_analysis(data):
    st.header("📊 Forest Data Analysis")
    
    df = pd.DataFrame(data)
    
    st.subheader("📋 Forest Growth Data")
    st.dataframe(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(df, x='year', y='ndvi', title='NDVI Trend Over Time', markers=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(df, x='year', y='forest_cover', title='Forest Cover Over Time')
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.subheader("📈 Statistical Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean NDVI", f"{df['ndvi'].mean():.3f}")
    with col2:
        st.metric("NDVI Std Dev", f"{df['ndvi'].std():.3f}")
    with col3:
        st.metric("Total Growth", f"{(df['ndvi'].iloc[-1] - df['ndvi'].iloc[0]):.3f}")
    with col4:
        st.metric("Years of Data", len(df))

def show_ml_predictions(data):
    st.header("🤖 Statistical Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Model Training")
        
        if st.session_state.trained:
            st.success("✅ Model is trained and ready!")
            model_info = st.session_state.ml_model.get_model_info()
            st.info(f"**Method:** {model_info['method']}")
            st.info(f"**Status:** Trained")
        else:
            st.warning("🔄 Model not trained yet")
        
        if st.button("🚀 Train Prediction Model", use_container_width=True):
            with st.spinner("Analyzing trends..."):
                results = st.session_state.ml_model.train(data)
                if results:
                    st.session_state.trained = True
                    st.rerun()
    
    with col2:
        st.subheader("🔮 Future Predictions")
        years_ahead = st.slider("Years to Predict", 1, 5, 3)
        
        if st.button("📈 Predict Future Growth", use_container_width=True):
            if st.session_state.trained:
                with st.spinner("Generating predictions..."):
                    predictions = st.session_state.ml_model.predict_future(data, years_ahead)
                    if predictions is not None:
                        # Display predictions
                        df_hist = pd.DataFrame(data)
                        future_years = list(range(df_hist['year'].max() + 1, 
                                                df_hist['year'].max() + 1 + years_ahead))
                        
                        # Create visualization
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df_hist['year'], y=df_hist['ndvi'],
                            name='Historical Data', mode='lines+markers',
                            line=dict(color='blue', width=3)
                        ))
                        fig.add_trace(go.Scatter(
                            x=future_years, y=predictions,
                            name='Predictions', mode='lines+markers',
                            line=dict(color='red', width=3, dash='dash')
                        ))
                        fig.update_layout(
                            title='Forest Growth Predictions',
                            xaxis_title='Year',
                            yaxis_title='NDVI'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Prediction details
                        st.subheader("📋 Prediction Details")
                        pred_df = pd.DataFrame({
                            'Year': future_years,
                            'Predicted NDVI': predictions,
                            'Growth %': ((predictions - df_hist['ndvi'].iloc[-1]) / df_hist['ndvi'].iloc[-1] * 100)
                        })
                        st.dataframe(pred_df.round(4))
            else:
                st.warning("Please train the model first")

if __name__ == "__main__":
    main()