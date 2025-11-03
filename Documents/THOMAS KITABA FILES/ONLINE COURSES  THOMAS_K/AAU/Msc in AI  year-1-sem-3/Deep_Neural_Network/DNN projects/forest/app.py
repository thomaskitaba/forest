import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import time

# Page configuration
st.set_page_config(
    page_title="Forest Growth Monitor - Enhanced CNN-LSTM",
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

# [Keep the same mock data and functions as before...]

def initialize_session_state():
    """Initialize required keys in streamlit session_state safely."""
    if 'ml_model' not in st.session_state:
        try:
            st.session_state.ml_model = ForestGrowthPredictor()
        except Exception as e:
            # If model instantiation fails, store None and report later
            st.session_state.ml_model = None
            st.error(f"Model initialization error: {e}")

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


def get_forest_data(geometry):
    """Return demo forest data (mock)."""
    return [
        {'year': 2018, 'ndvi': 0.65, 'forest_cover': 56, 'precipitation': 1200},
        {'year': 2019, 'ndvi': 0.68, 'forest_cover': 58, 'precipitation': 1250},
        {'year': 2020, 'ndvi': 0.71, 'forest_cover': 62, 'precipitation': 1180},
        {'year': 2021, 'ndvi': 0.69, 'forest_cover': 60, 'precipitation': 1150},
        {'year': 2022, 'ndvi': 0.73, 'forest_cover': 65, 'precipitation': 1300},
        {'year': 2023, 'ndvi': 0.75, 'forest_cover': 68, 'precipitation': 1280},
        {'year': 2024, 'ndvi': 0.77, 'forest_cover': 70, 'precipitation': 1350}
    ]


def show_dashboard(data):
    df = pd.DataFrame(data)
    st.header("🌿 Forest Growth Dashboard - Enhanced")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📍 Study Area")
        st.info("Demo Area")
        st.metric("Data Points", len(df))
    with col2:
        st.subheader("📈 Growth Metrics")
        current_ndvi = df['ndvi'].iloc[-1]
        ndvi_change = current_ndvi - df['ndvi'].iloc[0]
        st.metric("Current NDVI", f"{current_ndvi:.3f}", f"{ndvi_change:+.3f}")
    with col3:
        st.subheader("🎯 AI Status")
        st.metric("Trained", "✅" if st.session_state.trained else "❌")


def show_data_analysis(data):
    df = pd.DataFrame(data)
    st.header("📊 Data Analysis")
    st.dataframe(df)
    fig = px.line(df, x='year', y='ndvi', title='NDVI Trend', markers=True)
    st.plotly_chart(fig, use_container_width=True)


def display_enhanced_predictions(data, predictions, years_ahead):
    df_hist = pd.DataFrame(data)
    future_years = list(range(df_hist['year'].max() + 1, df_hist['year'].max() + 1 + years_ahead))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_hist['year'], y=df_hist['ndvi'], name='Historical', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=future_years, y=predictions, name='Predicted', mode='lines+markers'))
    st.plotly_chart(fig, use_container_width=True)


def show_model_insights(data):
    st.header("📈 Model Insights")
    results = st.session_state.get('ml_training_results')
    if results:
        st.write(results)
    else:
        st.info("No training results available. Train the model to see insights.")


def main():
    st.title("🌳 Forest Growth Monitoring System")
    st.markdown("### Enhanced CNN-LSTM with Adam, Cross-Validation & Comprehensive Evaluation")
    
    # Initialize session state
    initialize_session_state()
    
    # Enhanced sidebar with ML parameters
    st.sidebar.title("🧠 Enhanced ML Configuration")
    
    # Model parameters
    st.sidebar.subheader("Model Hyperparameters")
    sequence_length = st.sidebar.slider("Sequence Length", 3, 8, 5)
    epochs = st.sidebar.slider("Training Epochs", 50, 500, 100)
    use_cross_validation = st.sidebar.checkbox("Use Cross-Validation", value=True)
    
    # Update model parameters
    st.session_state.ml_model.sequence_length = sequence_length
    
    # App mode selection
    app_mode = st.sidebar.selectbox(
        "Choose Mode", 
        ["🏠 Dashboard", "📊 Data Analysis", "🤖 Enhanced Training", "📈 Model Insights", "⚙️ ML Configuration"]
    )
    
    # Demo data
    demo_data = get_forest_data(None)
    
    if app_mode == "🏠 Dashboard":
        show_dashboard(demo_data)
    elif app_mode == "📊 Data Analysis":
        show_data_analysis(demo_data)
    elif app_mode == "🤖 Enhanced Training":
        show_enhanced_training(demo_data, epochs, use_cross_validation)
    elif app_mode == "📈 Model Insights":
        show_model_insights(demo_data)
    elif app_mode == "⚙️ ML Configuration":
        show_ml_configuration()

def show_enhanced_training(data, epochs, use_cv):
    st.header("🤖 Enhanced CNN-LSTM Training")
    
    st.markdown("""
    **Advanced Features Included:**
    - 🎯 **Adam Optimization**: Adaptive learning rates with momentum
    - 📊 **Cross-Validation**: Time-series aware model validation  
    - 📈 **Comprehensive Metrics**: 7+ evaluation metrics
    - 🔧 **Weight Initialization**: He & Xavier initialization schemes
    - ⚡ **Activation Functions**: ReLU, Tanh, Leaky ReLU, Sigmoid
    - 🔍 **Feature Importance**: Permutation-based importance scores
    """)
    
    if st.session_state.training_in_progress:
        train_enhanced_model(data, epochs, use_cv)
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧠 Model Training")
        
        if st.session_state.trained:
            st.success("✅ Enhanced CNN-LSTM is trained!")
            model_info = st.session_state.ml_model.get_model_info()
            
            st.info(f"**Architecture:** {model_info['architecture']}")
            st.info(f"**Optimization:** {model_info['optimization']}")
            st.info(f"**Features:** {model_info['features_used']} engineered features")
            
            if 'cv_performance' in model_info:
                cv_info = model_info['cv_performance']
                st.info(f"**CV R²:** {cv_info['mean_r2']:.3f} ({cv_info['cv_folds']} folds)")
        
        if not st.session_state.trained and st.button("🚀 Train Enhanced Model", use_container_width=True, type="primary"):
            st.session_state.training_in_progress = True
            st.session_state.training_epochs = epochs
            st.session_state.use_cv = use_cv
            st.rerun()
    
    with col2:
        st.subheader("⚙️ Training Configuration")
        seq_len_display = getattr(st.session_state.get('ml_model'), 'sequence_length', 'N/A') if st.session_state.get('ml_model') is not None else 'N/A'
        st.metric("Sequence Length", seq_len_display)
        st.metric("Training Epochs", epochs)
        st.metric("Cross-Validation", "Enabled" if use_cv else "Disabled")
        st.metric("Optimizer", "Adam")
        
        st.subheader("🔮 Predictions")
        years_ahead = st.slider("Prediction Horizon", 1, 10, 3, key="enhanced_pred")
        
        if st.session_state.trained and st.button("📈 Generate Enhanced Predictions", use_container_width=True):
            with st.spinner("Generating predictions with enhanced model..."):
                predictions = st.session_state.ml_model.predict_future(data, years_ahead)
                if predictions is not None:
                    st.session_state.predictions = predictions
                    st.session_state.prediction_years = years_ahead
                    st.success("Enhanced predictions generated!")
        
        if st.session_state.predictions is not None and st.session_state.trained:
            display_enhanced_predictions(data, st.session_state.predictions, st.session_state.prediction_years)

def train_enhanced_model(data, epochs, use_cv):
    """Enhanced training with progress tracking"""
    st.header("🧠 Enhanced CNN-LSTM Training in Progress")
    
    progress_container = st.container()
    
    with progress_container:
        st.info("🚀 Starting enhanced training with Adam optimization...")
        
        # Training configuration display
        config_col1, config_col2, config_col3 = st.columns(3)
        with config_col1:
            st.metric("Epochs", epochs)
        with config_col2:
            st.metric("Optimizer", "Adam")
        with config_col3:
            st.metric("CV", "Yes" if use_cv else "No")
        
        # Simulate initial setup
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔄 Initializing weights with He/Xavier initialization...")
        time.sleep(1)
        progress_bar.progress(20)
        
        status_text.text("🔍 Preparing cross-validation splits..." if use_cv else "📊 Preparing training data...")
        time.sleep(1)
        progress_bar.progress(40)
        
        # Actual training
        try:
            status_text.text("🎯 Training with Adam optimization...")
            results = st.session_state.ml_model.train(data, epochs=epochs, use_cv=use_cv)
            
            if results:
                st.session_state.trained = True
                st.session_state.ml_training_results = results
                st.session_state.training_in_progress = False
                progress_bar.progress(100)
                status_text.text("✅ Enhanced training completed!")
                time.sleep(2)
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

def show_ml_configuration():
    st.header("⚙️ Machine Learning Configuration")
    
    st.markdown("""
    ### 🎯 Enhanced CNN-LSTM Architecture
    
    **Optimization & Training:**
    - **Adam Optimizer**: Combines RMSProp and Momentum with bias correction
    - **Learning Rate**: 0.001 with adaptive moment estimation
    - **Weight Initialization**: He for ReLU, Xavier for Tanh/Sigmoid
    
    **Model Architecture:**
    - **CNN Layers**: Feature extraction with ReLU activation
    - **LSTM Layers**: Temporal modeling with Tanh activation  
    - **Output Layer**: Linear activation for regression
    
    **Evaluation & Validation:**
    - **Time-Series Cross-Validation**: 5-fold CV respecting temporal order
    - **Comprehensive Metrics**: 7+ evaluation metrics
    - **Feature Importance**: Permutation-based importance scores
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Activation Functions")
        st.info("""
        **ReLU**: 
        - Used in CNN layers
        - Prevents vanishing gradient
        - Computationally efficient
        
        **Tanh**:
        - Used in LSTM layers  
        - Output range [-1, 1]
        - Zero-centered
        
        **Sigmoid**:
        - Available for gating mechanisms
        - Output range [0, 1]
        """)
    
    with col2:
        st.subheader("🔧 Optimization Details")
        st.info("""
        **Adam Optimizer**:
        - β₁ = 0.9 (First moment decay)
        - β₂ = 0.999 (Second moment decay)
        - ε = 1e-8 (Numerical stability)
        - Adaptive learning rates per parameter
        
        **Weight Initialization**:
        - He: sqrt(2/fan_in) for ReLU
        - Xavier: sqrt(1/fan_in) for Tanh
        """)
    
    st.subheader("📈 Evaluation Metrics")
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("R²", "Variance Explained")
        st.metric("MAE", "Mean Absolute Error")
    with metric_cols[1]:
        st.metric("RMSE", "Root Mean Squared Error")
        st.metric("MAPE", "Mean Absolute % Error")
    with metric_cols[2]:
        st.metric("Explained Var", "Explained Variance")
        st.metric("Dir Accuracy", "Direction Prediction")
    with metric_cols[3]:
        st.metric("Max Error", "Worst Case Error")
        st.metric("CV Score", "Cross-Validation")

# [Keep other functions the same as previous stable version...]

if __name__ == "__main__":
    main()