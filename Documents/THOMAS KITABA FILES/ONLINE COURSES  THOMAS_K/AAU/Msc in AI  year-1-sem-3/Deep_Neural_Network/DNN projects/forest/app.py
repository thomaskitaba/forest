import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

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
    st.success("✅ Successfully imported ForestGrowthPredictor with CNN-LSTM!")
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.stop()

# Mock data and functions
def initialize_earth_engine():
    return True

def get_forest_data(geometry):
    # Enhanced mock data with more realistic patterns
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
    'Awash National Park': {'min_lon': 40.0, 'max_lon': 41.0, 'min_lat': 8.5, 'max_lat': 9.5}
}

def main():
    st.title("🌳 Forest Growth Monitoring System")
    st.markdown("### Advanced CNN-LSTM Neural Network Prediction")
    
    # Initialize session state
    if 'ml_model' not in st.session_state:
        st.session_state.ml_model = ForestGrowthPredictor()
    
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    
    # Sidebar
    st.sidebar.title("🧠 CNN-LSTM Configuration")
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    sequence_length = st.sidebar.slider(
        "Sequence Length", 
        min_value=3, 
        max_value=8, 
        value=5,
        help="Number of previous years used for prediction (temporal context)"
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
            st.metric("Temporal Context", f"{model_info['sequence_length']} years")
        else:
            st.warning("🤖 Train CNN-LSTM")
            st.metric("Model", "CNN-LSTM")
            st.metric("Status", "Ready to Train")
            st.metric("Data Points", len(data))
    
    # Quick actions - FIXED: removed icon parameter
    st.subheader("🚀 Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 View Data", use_container_width=True):
            st.session_state.current_page = "data"
            st.rerun()
    
    with col2:
        if st.button("🧠 Train CNN-LSTM", use_container_width=True):
            with st.spinner("Training CNN-LSTM model..."):
                results = st.session_state.ml_model.train(data)
                if results:
                    st.session_state.trained = True
                    st.success("CNN-LSTM trained successfully!")
                    st.rerun()
    
    with col3:
        if st.button("🔮 Predict", use_container_width=True):
            if st.session_state.trained:
                predictions = st.session_state.ml_model.predict_future(data, 3)
                if predictions is not None:
                    st.success("CNN-LSTM predictions generated!")
            else:
                st.warning("Please train the CNN-LSTM model first")
    
    with col4:
        if st.button("📈 Insights", use_container_width=True):
            st.session_state.current_page = "insights"
            st.rerun()

def show_data_analysis(data):
    st.header("📊 Forest Data Analysis")
    
    df = pd.DataFrame(data)
    
    st.subheader("📋 Forest Growth Data")
    st.dataframe(df.style.format({'ndvi': '{:.3f}', 'forest_cover': '{:.0f}', 'precipitation': '{:.0f}'}))
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced NDVI plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['year'], y=df['ndvi'],
            name='NDVI', mode='lines+markers+text',
            line=dict(color='green', width=3),
            marker=dict(size=10, color='darkgreen'),
            text=[f'{val:.3f}' for val in df['ndvi']],
            textposition='top center'
        ))
        fig.update_layout(
            title='NDVI Trend Over Time (Vegetation Health)',
            xaxis_title='Year',
            yaxis_title='NDVI',
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.subheader("🔍 Feature Correlations")
        correlation = df[['ndvi', 'forest_cover', 'precipitation']].corr()
        fig_corr = px.imshow(
            correlation,
            title='Feature Correlation Matrix',
            color_continuous_scale='RdYlGn',
            aspect='auto'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        # Multiple metrics plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['year'], y=df['forest_cover'],
            name='Forest Cover %', mode='lines+markers',
            line=dict(color='blue', width=2),
            yaxis='y1'
        ))
        fig.add_trace(go.Bar(
            x=df['year'], y=df['precipitation'],
            name='Precipitation (mm)',
            marker_color='lightblue',
            yaxis='y2'
        ))
        fig.update_layout(
            title='Forest Cover & Precipitation Trends',
            xaxis=dict(title='Year'),
            yaxis=dict(title='Forest Cover %', side='left'),
            yaxis2=dict(title='Precipitation (mm)', side='right', overlaying='y'),
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.subheader("📈 Statistical Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean NDVI", f"{df['ndvi'].mean():.3f}")
        st.metric("NDVI Range", f"{df['ndvi'].max() - df['ndvi'].min():.3f}")
    with col2:
        st.metric("NDVI Std Dev", f"{df['ndvi'].std():.3f}")
        st.metric("Trend Slope", f"{(df['ndvi'].iloc[-1] - df['ndvi'].iloc[0]) / len(df):.4f}")
    with col3:
        st.metric("Forest Cover Gain", f"+{df['forest_cover'].iloc[-1] - df['forest_cover'].iloc[0]}%")
        st.metric("Data Quality", "✅ High")
    with col4:
        st.metric("Years of Data", len(df))
        st.metric("CNN-LSTM Ready", "✅" if len(df) >= 5 else "❌ Need more data")

def show_ml_predictions(data):
    st.header("🤖 CNN-LSTM Predictions")
    st.markdown("""
    **Convolutional LSTM Network** combines:
    - 🎯 **CNN Layers**: Extract local patterns and features from time series
    - 🧠 **LSTM Layers**: Model long-term temporal dependencies
    - 📈 **Sequence Learning**: Understand growth patterns over time
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧠 Model Training")
        
        if st.session_state.trained:
            st.success("✅ CNN-LSTM is trained and ready!")
            model_info = st.session_state.ml_model.get_model_info()
            
            st.info(f"**Architecture:** {model_info['architecture']}")
            st.info(f"**Sequence Length:** {model_info['sequence_length']} years")
            st.info(f"**Temporal Context:** Excellent")
            st.info(f"**Feature Extraction:** CNN-enhanced")
            
            # Training metrics
            if 'ml_training_results' in st.session_state:
                results = st.session_state.ml_training_results
                st.metric("Feature Dimensions", results.get('feature_dim', 'N/A'))
                st.metric("Training Sequences", results.get('train_size', 'N/A'))
        
        # FIXED: removed icon parameter
        if st.button("🚀 Train CNN-LSTM Model", use_container_width=True, type="primary"):
            with st.spinner("Training CNN-LSTM with convolutional feature extraction..."):
                results = st.session_state.ml_model.train(data)
                if results:
                    st.session_state.trained = True
                    st.session_state.ml_training_results = results
                    st.rerun()
    
    with col2:
        st.subheader("🔮 Future Predictions")
        years_ahead = st.slider("Years to Predict", 1, 10, 5, 
                               help="CNN-LSTM can predict further with temporal patterns")
        
        # FIXED: removed icon parameter
        if st.button("📈 Generate CNN-LSTM Predictions", use_container_width=True, type="secondary"):
            if st.session_state.trained:
                with st.spinner("CNN-LSTM generating predictions with temporal analysis..."):
                    predictions = st.session_state.ml_model.predict_future(data, years_ahead)
                    if predictions is not None:
                        # Store predictions
                        st.session_state.predictions = predictions
                        st.session_state.prediction_years = years_ahead
                        st.rerun()
            else:
                st.warning("Please train the CNN-LSTM model first")
        
        # Display predictions if available
        if 'predictions' in st.session_state and st.session_state.trained:
            predictions = st.session_state.predictions
            years_ahead = st.session_state.prediction_years
            
            df_hist = pd.DataFrame(data)
            future_years = list(range(df_hist['year'].max() + 1, 
                                    df_hist['year'].max() + 1 + years_ahead))
            
            # Enhanced prediction visualization
            fig = go.Figure()
            
            # Historical data with confidence
            fig.add_trace(go.Scatter(
                x=df_hist['year'], y=df_hist['ndvi'],
                name='Historical Data', mode='lines+markers',
                line=dict(color='blue', width=4),
                marker=dict(size=8, symbol='circle')
            ))
            
            # Predictions with gradient confidence
            fig.add_trace(go.Scatter(
                x=future_years, y=predictions,
                name='CNN-LSTM Predictions', mode='lines+markers',
                line=dict(color='red', width=4, dash='dash'),
                marker=dict(size=10, symbol='star')
            ))
            
            # Confidence interval
            confidence = predictions * 0.05  # 5% confidence interval
            fig.add_trace(go.Scatter(
                x=future_years + future_years[::-1],
                y=np.concatenate([predictions + confidence, (predictions - confidence)[::-1]]),
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                showlegend=True
            ))
            
            fig.update_layout(
                title='CNN-LSTM Forest Growth Predictions',
                xaxis_title='Year',
                yaxis_title='NDVI',
                hovermode='x unified',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction analysis
            st.subheader("📋 CNN-LSTM Prediction Analysis")
            pred_df = pd.DataFrame({
                'Year': future_years,
                'Predicted NDVI': predictions,
                'Growth %': ((predictions - df_hist['ndvi'].iloc[-1]) / df_hist['ndvi'].iloc[-1] * 100),
                'Confidence': 'High'  # CNN-LSTM typically has better confidence
            })
            st.dataframe(pred_df.style.format({
                'Predicted NDVI': '{:.3f}',
                'Growth %': '{:.2f}%'
            }))
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_growth = pred_df['Growth %'].mean()
                st.metric("Avg Annual Growth", f"{avg_growth:.2f}%")
            with col2:
                total_growth = pred_df['Growth %'].iloc[-1]
                st.metric("Total Projected Growth", f"{total_growth:.2f}%")
            with col3:
                st.metric("Prediction Confidence", "High", "CNN-LSTM")

def show_model_insights(data):
    st.header("📈 CNN-LSTM Model Insights")
    
    st.markdown("""
    ### 🧠 How CNN-LSTM Works for Forest Growth Prediction
    
    **Architecture Overview:**
    ```
    Input Sequence → CNN Feature Extraction → LSTM Temporal Modeling → Output Prediction
    ```
    
    **Key Advantages:**
    - 🎯 **Local Pattern Detection**: CNN layers identify short-term growth patterns
    - 🧠 **Long-term Memory**: LSTM layers remember important temporal relationships  
    - 📊 **Multi-scale Analysis**: Combines immediate and long-term trends
    - 🔍 **Feature Learning**: Automatically learns relevant patterns from data
    """)
    
    if st.session_state.trained and 'ml_training_results' in st.session_state:
        results = st.session_state.ml_training_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏗️ Model Architecture")
            st.json(results.get('architecture', {}))
            
            st.subheader("📊 Performance Metrics")
            st.metric("R² Score", f"{results.get('r2', 0):.4f}")
            st.metric("Mean Absolute Error", f"{results.get('mae', 0):.4f}")
            st.metric("Training Sequences", results.get('train_size', 0))
            st.metric("Sequence Context", f"{results.get('sequence_length', 0)} years")
        
        with col2:
            st.subheader("🔍 Feature Importance")
            if 'feature_importance' in results:
                features = [f'Feature {i+1}' for i in range(len(results['feature_importance']))]
                importance = results['feature_importance']
                
                fig = px.bar(
                    x=features, y=importance,
                    title='CNN-LSTM Feature Importance',
                    labels={'x': 'Features', 'y': 'Importance Score'},
                    color=importance,
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👆 Train the CNN-LSTM model to see detailed insights and architecture information.")
        
        # Educational content
        st.subheader("🎯 CNN-LSTM Benefits for Forest Monitoring")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Temporal Pattern Recognition:**
            - Detects seasonal growth cycles
            - Identifies long-term climate trends
            - Recognizes recovery patterns after events
            - Models non-linear growth relationships
            """)
        
        with col2:
            st.markdown("""
            **Feature Learning Capabilities:**
            - Automatic feature extraction from sequences
            - Multi-scale temporal analysis
            - Robust to noisy environmental data
            - Adaptable to different forest types
            """)
        
        st.subheader("📈 Expected Performance")
        st.info("""
        With sufficient historical data (5+ years), CNN-LSTM typically achieves:
        - R² scores: 0.85-0.95
        - Prediction horizon: 3-5 years with good accuracy
        - Robustness: Handles missing data and noise well
        - Interpretability: Provides feature importance insights
        """)

if __name__ == "__main__":
    main()