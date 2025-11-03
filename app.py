import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import time

# Page configuration - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Forest Growth Monitor - CNN-LSTM",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

try:
    from ml_utils import ForestGrowthPredictor
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.stop()

# Mock data and functions
def initialize_earth_engine():
    return True

def get_forest_data(geometry):
    return [
        {'year': 2018, 'ndvi': 0.65, 'forest_cover': 56, 'precipitation': 1200},
        {'year': 2019, 'ndvi': 0.68, 'forest_cover': 58, 'precipitation': 1250},
        {'year': 2020, 'ndvi': 0.71, 'forest_cover': 62, 'precipitation': 1180},
        {'year': 2021, 'ndvi': 0.69, 'forest_cover': 60, 'precipitation': 1150},
        {'year': 2022, 'ndvi': 0.73, 'forest_cover': 65, 'precipitation': 1300},
        {'year': 2023, 'ndvi': 0.75, 'forest_cover': 68, 'precipitation': 1280},
        {'year': 2024, 'ndvi': 0.77, 'forest_cover': 70, 'precipitation': 1350}
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
    'Simien Mountains': {'min_lon': 38.0, 'max_lon': 39.0, 'min_lat': 13.0, 'max_lat': 14.0},
}

def initialize_session_state():
    """Initialize all session state variables"""
    if 'ml_model' not in st.session_state:
        st.session_state.ml_model = ForestGrowthPredictor()
    
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    
    if 'training_in_progress' not in st.session_state:
        st.session_state.training_in_progress = False
    
    if 'ml_training_results' not in st.session_state:
        st.session_state.ml_training_results = None
    
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    
    if 'prediction_years' not in st.session_state:
        st.session_state.prediction_years = 3

def main():
    st.title("🌳 Forest Growth Monitoring System")
    st.markdown("### Advanced CNN-LSTM Neural Network Prediction")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    st.sidebar.title("🧠 CNN-LSTM Configuration")
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    sequence_length = st.sidebar.slider(
        "Sequence Length", 
        min_value=3, 
        max_value=8, 
        value=5,
        help="Number of previous years used for prediction"
    )
    
    # Update model sequence length
    st.session_state.ml_model.sequence_length = sequence_length
    
    # App mode selection
    app_mode = st.sidebar.selectbox(
        "Choose Mode", 
        ["🏠 Dashboard", "📊 Data Analysis", "🤖 CNN-LSTM Predictions", "📈 Model Insights"]
    )
    
    # AOI Selection
    st.sidebar.subheader("📍 Area Selection")
    aoi_names = ["Custom"] + list(AOIS.keys())
    selected_aoi = st.sidebar.selectbox("Select Area", aoi_names)
    
    # Demo data
    demo_data = get_forest_data(None)
    
    # Display based on selected mode
    if app_mode == "🏠 Dashboard":
        show_dashboard(demo_data)
    elif app_mode == "📊 Data Analysis":
        show_data_analysis(demo_data)
    elif app_mode == "🤖 CNN-LSTM Predictions":
        show_ml_predictions(demo_data)
    elif app_mode == "📈 Model Insights":
        show_model_insights(demo_data)

def show_dashboard(data):
    st.header("🌿 Forest Growth Dashboard - CNN-LSTM Enhanced")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📍 Study Area")
        st.info("Bale Mountains Region, Ethiopia")
        st.metric("Area Size", "1,200 km²")
        st.metric("Planting Started", "2018")
        st.metric("CNN-LSTM Ready", "✅" if st.session_state.trained else "❌")
    
    with col2:
        st.subheader("📈 Growth Metrics")
        df = pd.DataFrame(data)
        current_ndvi = df['ndvi'].iloc[-1]
        ndvi_change = current_ndvi - df['ndvi'].iloc[0]
        growth_rate = ((current_ndvi / df['ndvi'].iloc[0]) - 1) * 100
        
        st.metric("Current NDVI", f"{current_ndvi:.3f}", f"{ndvi_change:+.3f}")
        st.metric("Forest Cover", f"{df['forest_cover'].iloc[-1]}%", "+12%")
        st.metric("Annual Growth Rate", f"{growth_rate:.1f}%")
    
    with col3:
        st.subheader("🎯 AI Analysis")
        if st.session_state.trained:
            st.success("✅ CNN-LSTM Trained")
            model_info = st.session_state.ml_model.get_model_info()
            st.metric("Sequence Length", model_info['sequence_length'])
            st.metric("Architecture", "CNN-LSTM")
            st.metric("Status", "Ready")
        else:
            st.warning("🤖 Train CNN-LSTM")
            st.metric("Model", "CNN-LSTM")
            st.metric("Status", "Ready to Train")
            st.metric("Data Points", len(data))
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 View Data Analysis", use_container_width=True):
            st.session_state.current_page = "data"
    
    with col2:
        if st.button("🧠 Train Model", use_container_width=True) and not st.session_state.training_in_progress:
            st.session_state.training_in_progress = True
            st.rerun()
    
    with col3:
        if st.button("🔮 Make Predictions", use_container_width=True):
            st.session_state.current_page = "predictions"

def show_data_analysis(data):
    st.header("📊 Forest Data Analysis")
    
    df = pd.DataFrame(data)
    
    st.subheader("📋 Forest Growth Data")
    st.dataframe(df.style.format({'ndvi': '{:.3f}', 'forest_cover': '{:.0f}', 'precipitation': '{:.0f}'}))
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(df, x='year', y='ndvi', title='NDVI Trend Over Time', markers=True)
        fig.update_layout(yaxis_title="NDVI", xaxis_title="Year")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(df, x='year', y='forest_cover', title='Forest Cover Over Time')
        fig.update_layout(yaxis_title="Forest Cover %", xaxis_title="Year")
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
    st.header("🤖 CNN-LSTM Predictions")
    
    # Handle training if in progress
    if st.session_state.training_in_progress:
        train_model(data)
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧠 Model Training")
        
        if st.session_state.trained:
            st.success("✅ CNN-LSTM is trained and ready!")
            model_info = st.session_state.ml_model.get_model_info()
            
            st.info(f"**Architecture:** {model_info['architecture']}")
            st.info(f"**Sequence Length:** {model_info['sequence_length']} years")
            st.info(f"**Status:** Ready for predictions")
            
            if st.session_state.ml_training_results:
                results = st.session_state.ml_training_results
                st.metric("R² Score", f"{results.get('r2', 0):.3f}")
                st.metric("Training Sequences", results.get('train_size', 0))
        
        if not st.session_state.trained and st.button("🚀 Train CNN-LSTM Model", use_container_width=True, type="primary"):
            st.session_state.training_in_progress = True
            st.rerun()
    
    with col2:
        st.subheader("🔮 Future Predictions")
        years_ahead = st.slider("Years to Predict", 1, 10, 3)
        
        if st.session_state.trained and st.button("📈 Generate Predictions", use_container_width=True, type="secondary"):
            with st.spinner("Generating predictions..."):
                predictions = st.session_state.ml_model.predict_future(data, years_ahead)
                if predictions is not None:
                    st.session_state.predictions = predictions
                    st.session_state.prediction_years = years_ahead
                    st.success("Predictions generated successfully!")
        
        # Display predictions if available
        if st.session_state.predictions is not None and st.session_state.trained:
            display_predictions(data, st.session_state.predictions, st.session_state.prediction_years)

def train_model(data):
    """Handle model training in a separate function to avoid flickering"""
    st.header("🧠 Training CNN-LSTM Model")
    
    # Create a progress container
    progress_container = st.container()
    
    with progress_container:
        st.info("🔄 Starting CNN-LSTM training...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate progress updates
        for i in range(5):
            progress = (i + 1) * 20
            progress_bar.progress(progress)
            if i == 0:
                status_text.text("🔄 Preparing data...")
            elif i == 1:
                status_text.text("🔍 Extracting features with CNN...")
            elif i == 2:
                status_text.text("🧠 Modeling temporal patterns with LSTM...")
            elif i == 3:
                status_text.text("📊 Calculating performance metrics...")
            else:
                status_text.text("✅ Finalizing model...")
            time.sleep(0.5)  # Simulate work
        
        # Actual training
        try:
            status_text.text("🎯 Training model...")
            results = st.session_state.ml_model.train(data)
            
            if results:
                st.session_state.trained = True
                st.session_state.ml_training_results = results
                st.session_state.training_in_progress = False
                progress_bar.progress(100)
                status_text.text("✅ Training completed successfully!")
                time.sleep(1)  # Show success message briefly
                st.rerun()
            else:
                st.session_state.training_in_progress = False
                progress_bar.progress(0)
                status_text.text("❌ Training failed")
                if st.button("🔄 Try Again"):
                    st.rerun()
                
        except Exception as e:
            st.session_state.training_in_progress = False
            progress_bar.progress(0)
            status_text.text(f"❌ Training error: {str(e)}")
            if st.button("🔄 Try Again"):
                st.rerun()

def display_predictions(data, predictions, years_ahead):
    """Display predictions in a stable way"""
    df_hist = pd.DataFrame(data)
    future_years = list(range(df_hist['year'].max() + 1, 
                            df_hist['year'].max() + 1 + years_ahead))
    
    # Create visualization
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=df_hist['year'], y=df_hist['ndvi'],
        name='Historical Data', mode='lines+markers',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))
    
    # Predictions
    fig.add_trace(go.Scatter(
        x=future_years, y=predictions,
        name='CNN-LSTM Predictions', mode='lines+markers',
        line=dict(color='red', width=3, dash='dash'),
        marker=dict(size=10, symbol='star')
    ))
    
    fig.update_layout(
        title='CNN-LSTM Forest Growth Predictions',
        xaxis_title='Year',
        yaxis_title='NDVI',
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction analysis
    st.subheader("📋 Prediction Analysis")
    pred_df = pd.DataFrame({
        'Year': future_years,
        'Predicted NDVI': predictions,
        'Growth %': ((predictions - df_hist['ndvi'].iloc[-1]) / df_hist['ndvi'].iloc[-1] * 100)
    })
    st.dataframe(pred_df.style.format({
        'Predicted NDVI': '{:.3f}',
        'Growth %': '{:.2f}%'
    }))

def show_model_insights(data):
    st.header("📈 CNN-LSTM Model Insights")
    
    if st.session_state.trained and st.session_state.ml_training_results:
        results = st.session_state.ml_training_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Performance Metrics")
            st.metric("R² Score", f"{results.get('r2', 0):.4f}")
            st.metric("Mean Absolute Error", f"{results.get('mae', 0):.4f}")
            st.metric("Training Sequences", results.get('train_size', 0))
            st.metric("Sequence Length", results.get('sequence_length', 0))
        
        with col2:
            st.subheader("🔍 Feature Importance")
            if 'feature_importance' in results and results['feature_importance']:
                features = [f'F{i+1}' for i in range(len(results['feature_importance']))]
                fig = px.bar(
                    x=features, y=results['feature_importance'],
                    title='Feature Importance Scores',
                    labels={'x': 'Features', 'y': 'Importance'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No feature importance data available")
    
    else:
        st.info("👆 Train the CNN-LSTM model to see detailed insights")
        
        st.subheader("🎯 CNN-LSTM Benefits")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Temporal Pattern Recognition:**
            - Detects seasonal growth cycles
            - Identifies long-term climate trends
            - Models non-linear growth relationships
            """)
        
        with col2:
            st.markdown("""
            **Feature Learning:**
            - Automatic feature extraction
            - Multi-scale temporal analysis
            - Robust to noisy data
            """)

if __name__ == "__main__":
    main()