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

def get_forest_data(geometry, aoi_name=None):
    """Return mock forest data. If geometry (or named AOI) is provided, slightly vary the base NDVI
    to simulate different areas. In production this would query Earth Engine or a database.
    """
    # Base mock series
    base = [0.65, 0.68, 0.71, 0.69, 0.73, 0.75, 0.77]

    # If aoi_name wasn't provided (older call-sites), try session state
    if aoi_name is None:
        try:
            aoi_name = st.session_state.get('selected_aoi')
        except Exception:
            aoi_name = None

    # Determine a deterministic offset based on AOI name or geometry
    offset = 0.0
    if aoi_name and aoi_name in AOIS:
        # small offsets per AOI to simulate variation
        offsets_map = {name: (i + 1) * 0.01 for i, name in enumerate(AOIS.keys())}
        offset = offsets_map.get(aoi_name, 0.0)
    elif geometry is not None:
        # use min_lon to create a repeatable offset
        try:
            min_lon = geometry['coordinates'][0][0][0]
            offset = (abs(min_lon) % 1) * 0.05
        except Exception:
            offset = 0.0

    years = list(range(2018, 2018 + len(base)))
    data = []
    for i, y in enumerate(years):
        ndvi = base[i] + offset
        data.append({
            'year': y,
            'ndvi': round(ndvi, 3),
            'forest_cover': 50 + int((ndvi - 0.6) * 100),
            'precipitation': 1100 + int((ndvi - 0.6) * 1000)
        })

    return data

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
    # AOI selection state
    if 'selected_aoi' not in st.session_state:
        st.session_state.selected_aoi = 'Custom'
    if 'aoi_geometry' not in st.session_state:
        st.session_state.aoi_geometry = None

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

    # Save selection in session state so all pages can access it
    st.session_state.selected_aoi = selected_aoi

    # Compute geometry for named AOIs
    if selected_aoi != 'Custom' and selected_aoi in AOIS:
        a = AOIS[selected_aoi]
        st.session_state.aoi_geometry = create_aoi_geometry(a['min_lon'], a['max_lon'], a['min_lat'], a['max_lat'])
    else:
        st.session_state.aoi_geometry = None

    # Demo data for the selected AOI (geometry may be None for Custom)
    # Call with a single arg for compatibility with other modules / older signatures
    demo_data = get_forest_data(st.session_state.aoi_geometry)
    
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
        # Show currently selected AOI
        aoi_label = st.session_state.get('selected_aoi', 'Custom')
        st.info(aoi_label)
        # If a known AOI is selected, show a simple area estimate
        if aoi_label in AOIS:
            a = AOIS[aoi_label]
            # crude area estimate (degrees -> approx km)
            area_deg = abs(a['max_lon'] - a['min_lon']) * abs(a['max_lat'] - a['min_lat'])
            # 1 deg ~ 111 km, square -> 111^2
            area_km2 = int(area_deg * 111 * 111)
            st.metric("Area Size", f"{area_km2:,} km²")
        else:
            st.metric("Area Size", "Custom")
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
    aoi_label = st.session_state.get('selected_aoi', 'Custom')
    st.header(f"📊 Forest Data Analysis — {aoi_label}")
    
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
            # Safely handle feature importance which may be a numpy array or list.
            fi = results.get('feature_importance')
            if fi is not None and np.array(fi).size > 0:
                # ensure a flat array for plotting
                fi_arr = np.array(fi).ravel()
                # Prefer human-readable names from the training results
                fname_list = results.get('feature_names') if results is not None else None
                if fname_list and len(fname_list) == len(fi_arr):
                    features = fname_list
                else:
                    features = [f'F{i+1}' for i in range(len(fi_arr))]

                fig = px.bar(
                    x=features, y=fi_arr,
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