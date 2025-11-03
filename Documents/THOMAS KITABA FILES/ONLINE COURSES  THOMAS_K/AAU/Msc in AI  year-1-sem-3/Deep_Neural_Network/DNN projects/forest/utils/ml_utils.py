import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import math

warnings.filterwarnings('ignore')

class CNNLSTMModel:
    """
    Enhanced CNN-LSTM model with Adam optimization, cross-validation, 
    and comprehensive evaluation metrics
    """
    
    def __init__(self, sequence_length=5, hidden_units=64, learning_rate=0.001):
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.weights = {}
        self.biases = {}
        self.is_trained = False
        self.training_history = []
        
    def he_initialization(self, shape: Tuple[int, int]) -> np.ndarray:
        """He initialization for ReLU activation"""
        return np.random.randn(*shape) * np.sqrt(2.0 / shape[0])
    
    def xavier_initialization(self, shape: Tuple[int, int]) -> np.ndarray:
        """Xavier/Glorot initialization for tanh/sigmoid"""
        return np.random.randn(*shape) * np.sqrt(1.0 / shape[0])
    
    def initialize_weights(self, input_dim: int):
        """Initialize all weights with proper initialization schemes"""
        # CNN-like feature extraction weights
        self.weights['conv1'] = self.he_initialization((input_dim, self.hidden_units))
        self.weights['conv2'] = self.he_initialization((self.hidden_units, self.hidden_units // 2))
        
        # LSTM-like weights
        self.weights['lstm_input'] = self.xavier_initialization((self.hidden_units // 2, self.hidden_units))
        self.weights['lstm_hidden'] = self.xavier_initialization((self.hidden_units, self.hidden_units))
        
        # Output layer
        self.weights['output'] = self.xavier_initialization((self.hidden_units, 1))
        
        # Biases
        self.biases['conv1'] = np.zeros((1, self.hidden_units))
        self.biases['conv2'] = np.zeros((1, self.hidden_units // 2))
        self.biases['lstm'] = np.zeros((1, self.hidden_units))
        self.biases['output'] = np.zeros((1, 1))
        
        # Adam optimizer parameters
        self.m = {}  # First moment vector
        self.v = {}  # Second moment vector
        for key in self.weights:
            self.m[key] = np.zeros_like(self.weights[key])
            self.v[key] = np.zeros_like(self.weights[key])
        for key in self.biases:
            self.m[f'b_{key}'] = np.zeros_like(self.biases[key])
            self.v[f'b_{key}'] = np.zeros_like(self.biases[key])
        
        self.t = 0  # Time step for Adam
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    def tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh activation function"""
        return np.tanh(x)
    
    def tanh_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of tanh"""
        return 1 - np.tanh(x) ** 2
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid"""
        sig = self.sigmoid(x)
        return sig * (1 - sig)
    
    def leaky_relu(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU activation function"""
        return np.where(x > 0, x, alpha * x)
    
    def leaky_relu_derivative(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Derivative of Leaky ReLU"""
        return np.where(x > 0, 1, alpha)
    
    def cnn_forward(self, X: np.ndarray) -> np.ndarray:
        """CNN-like forward pass for feature extraction"""
        # First convolutional layer with ReLU
        conv1 = X @ self.weights['conv1'] + self.biases['conv1']
        conv1_activated = self.relu(conv1)
        
        # Second convolutional layer with ReLU
        conv2 = conv1_activated @ self.weights['conv2'] + self.biases['conv2']
        conv2_activated = self.relu(conv2)
        
        return conv2_activated
    
    def lstm_forward(self, features: np.ndarray, prev_hidden: np.ndarray = None) -> np.ndarray:
        """LSTM-like forward pass for sequence modeling"""
        if prev_hidden is None:
            batch_size = features.shape[0]
            prev_hidden = np.zeros((batch_size, self.hidden_units))
        
        # Simplified LSTM cell
        input_transform = features @ self.weights['lstm_input']
        hidden_transform = prev_hidden @ self.weights['lstm_hidden']
        
        # Combined gates (simplified)
        combined = input_transform + hidden_transform + self.biases['lstm']
        new_hidden = self.tanh(combined)
        
        return new_hidden
    
    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        """Complete forward pass through CNN-LSTM"""
        # CNN feature extraction
        cnn_features = self.cnn_forward(X)
        
        # LSTM sequence modeling
        lstm_output = self.lstm_forward(cnn_features)
        
        # Output layer with linear activation
        output = lstm_output @ self.weights['output'] + self.biases['output']
        
        return output.flatten()
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Mean Squared Error loss"""
        return np.mean((y_true - y_pred) ** 2)
    
    def backward_pass(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass for gradient computation"""
        batch_size = X.shape[0]
        
        # Output layer gradients
        d_output = 2 * (y_pred - y_true) / batch_size
        d_weights_output = self.lstm_hidden.T @ d_output.reshape(-1, 1)
        d_biases_output = np.sum(d_output, axis=0, keepdims=True)
        
        # LSTM gradients (simplified)
        d_lstm_hidden = d_output.reshape(-1, 1) @ self.weights['output'].T
        d_lstm = d_lstm_hidden * self.tanh_derivative(self.lstm_hidden)
        
        d_weights_lstm_input = self.cnn_features.T @ d_lstm
        d_weights_lstm_hidden = self.prev_hidden.T @ d_lstm
        d_biases_lstm = np.sum(d_lstm, axis=0, keepdims=True)
        
        # CNN gradients
        d_cnn = d_lstm @ self.weights['lstm_input'].T
        d_conv2 = d_cnn * self.relu_derivative(self.conv2)
        d_weights_conv2 = self.conv1_activated.T @ d_conv2
        d_biases_conv2 = np.sum(d_conv2, axis=0, keepdims=True)
        
        d_conv1 = d_conv2 @ self.weights['conv2'].T * self.relu_derivative(self.conv1)
        d_weights_conv1 = X.T @ d_conv1
        d_biases_conv1 = np.sum(d_conv1, axis=0, keepdims=True)
        
        gradients = {
            'conv1': d_weights_conv1, 'b_conv1': d_biases_conv1,
            'conv2': d_weights_conv2, 'b_conv2': d_biases_conv2,
            'lstm_input': d_weights_lstm_input, 'lstm_hidden': d_weights_lstm_hidden,
            'b_lstm': d_biases_lstm, 'output': d_weights_output, 'b_output': d_biases_output
        }
        
        return gradients
    
    def adam_optimizer(self, gradients: Dict[str, np.ndarray], beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        """Adam optimization algorithm"""
        self.t += 1
        
        for key in gradients:
            if key in self.weights:
                # Update first moment
                self.m[key] = beta1 * self.m[key] + (1 - beta1) * gradients[key]
                # Update second moment
                self.v[key] = beta2 * self.v[key] + (1 - beta2) * (gradients[key] ** 2)
                
                # Bias correction
                m_hat = self.m[key] / (1 - beta1 ** self.t)
                v_hat = self.v[key] / (1 - beta2 ** self.t)
                
                # Update weights
                self.weights[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            
            elif key.startswith('b_') and key[2:] in self.biases:
                bias_key = key[2:]
                # Update first moment
                self.m[key] = beta1 * self.m[key] + (1 - beta1) * gradients[key]
                # Update second moment
                self.v[key] = beta2 * self.v[key] + (1 - beta2) * (gradients[key] ** 2)
                
                # Bias correction
                m_hat = self.m[key] / (1 - beta1 ** self.t)
                v_hat = self.v[key] / (1 - beta2 ** self.t)
                
                # Update biases
                self.biases[bias_key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    
    def train_epoch(self, X: np.ndarray, y: np.ndarray) -> float:
        """Train for one epoch"""
        # Store intermediate values for backward pass
        self.conv1 = X @ self.weights['conv1'] + self.biases['conv1']
        self.conv1_activated = self.relu(self.conv1)
        
        self.conv2 = self.conv1_activated @ self.weights['conv2'] + self.biases['conv2']
        self.conv2_activated = self.relu(self.conv2)
        self.cnn_features = self.conv2_activated
        
        # Simplified LSTM (using current features only)
        self.prev_hidden = np.zeros((X.shape[0], self.hidden_units))
        lstm_input = self.cnn_features @ self.weights['lstm_input']
        lstm_hidden = self.prev_hidden @ self.weights['lstm_hidden']
        self.lstm_combined = lstm_input + lstm_hidden + self.biases['lstm']
        self.lstm_hidden = self.tanh(self.lstm_combined)
        
        # Output
        y_pred = self.lstm_hidden @ self.weights['output'] + self.biases['output']
        y_pred = y_pred.flatten()
        
        # Compute loss and gradients
        loss = self.compute_loss(y, y_pred)
        gradients = self.backward_pass(X, y, y_pred)
        
        # Update weights with Adam
        self.adam_optimizer(gradients)
        
        return loss
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict[str, List[float]]:
        """Time-series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = {
            'mse': [], 'mae': [], 'r2': [], 'rmse': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Initialize fresh model for each fold
            fold_model = CNNLSTMModel(self.sequence_length, self.hidden_units, self.learning_rate)
            fold_model.initialize_weights(X_train.shape[1])
            
            # Train fold model
            for epoch in range(50):  # Fewer epochs for CV
                fold_model.train_epoch(X_train, y_train)
            
            # Validate
            y_pred = fold_model.forward_pass(X_val)
            
            # Calculate metrics
            cv_scores['mse'].append(mean_squared_error(y_val, y_pred))
            cv_scores['mae'].append(mean_absolute_error(y_val, y_pred))
            cv_scores['r2'].append(r2_score(y_val, y_pred))
            cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_val, y_pred)))
        
        return cv_scores

class ForestGrowthPredictor:
    """
    Enhanced Forest Growth Predictor with comprehensive ML features
    """
    
    def __init__(self):
        self.is_trained = False
        self.sequence_length = 5
        self.model = None
        self.cv_results = None
        self.feature_names = None
        
    def prepare_data(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare enhanced features for CNN-LSTM"""
        df = pd.DataFrame(data)
        ndvi_values = df['ndvi'].values
        
        if len(ndvi_values) < self.sequence_length + 1:
            return np.array([]), np.array([]), []
        
        # Create sequences
        X, y = [], []
        for i in range(len(ndvi_values) - self.sequence_length):
            X.append(ndvi_values[i:(i + self.sequence_length)])
            y.append(ndvi_values[i + self.sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        if len(X) == 0:
            return np.array([]), np.array([]), []
        
        # Enhanced feature engineering
        X_enhanced = []
        feature_names = []
        
        for i, sequence in enumerate(X):
            features = []
            
            # Original sequence values
            features.extend(sequence)
            feature_names.extend([f'lag_{j+1}' for j in range(len(sequence))])
            
            # Statistical features
            stats_features = [
                np.mean(sequence), np.std(sequence), np.min(sequence), 
                np.max(sequence), np.median(sequence), sequence[-1] - sequence[0]
            ]
            features.extend(stats_features)
            feature_names.extend(['mean', 'std', 'min', 'max', 'median', 'trend'])
            
            # Rolling features
            for window in [2, 3]:
                if len(sequence) >= window:
                    rolling_mean = np.convolve(sequence, np.ones(window)/window, mode='valid')
                    features.append(rolling_mean[-1])
                    feature_names.append(f'rolling_mean_{window}')
                    
                    if window > 1:
                        rolling_trend = sequence[-1] - sequence[-window]
                        features.append(rolling_trend)
                        feature_names.append(f'rolling_trend_{window}')
            
            # Ensure consistent feature size
            while len(features) < 20:
                features.append(0.0)
                feature_names.append(f'padding_{len(features)}')
            
            if len(features) > 20:
                features = features[:20]
                feature_names = feature_names[:20]
                
            X_enhanced.append(features)
        
        self.feature_names = feature_names[:20]  # Store feature names
        return np.array(X_enhanced), y, self.feature_names
    
    def comprehensive_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Comprehensive evaluation metrics"""
        metrics = {}
        
        # Regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100
        
        # Additional metrics
        metrics['explained_variance'] = 1 - np.var(y_true - y_pred) / np.var(y_true)
        metrics['max_error'] = np.max(np.abs(y_true - y_pred))
        
        # Direction accuracy
        direction_true = np.diff(y_true) > 0
        direction_pred = np.diff(y_pred) > 0
        if len(direction_true) > 0:
            metrics['direction_accuracy'] = np.mean(direction_true == direction_pred)
        else:
            metrics['direction_accuracy'] = 0.0
        
        return metrics
    
    def train(self, data: List[Dict], epochs: int = 100, use_cv: bool = True) -> Optional[Dict]:
        """Train with comprehensive features"""
        try:
            st.info("🔄 Preparing data for enhanced CNN-LSTM training...")
            
            # Prepare data
            X, y, feature_names = self.prepare_data(data)
            
            if len(X) == 0:
                st.error("❌ Not enough data for training")
                return None
            
            st.info(f"📊 Training on {len(X)} sequences with {X.shape[1]} features")
            
            # Initialize model
            self.model = CNNLSTMModel(self.sequence_length, hidden_units=64, learning_rate=0.001)
            self.model.initialize_weights(X.shape[1])
            
            # Cross-validation
            if use_cv and len(X) >= 10:
                st.info("🔍 Performing time-series cross-validation...")
                self.cv_results = self.model.cross_validate(X, y, n_splits=min(5, len(X)//2))
                
                # Display CV results
                cv_col1, cv_col2, cv_col3, cv_col4 = st.columns(4)
                with cv_col1:
                    st.metric("CV R²", f"{np.mean(self.cv_results['r2']):.3f} ± {np.std(self.cv_results['r2']):.3f}")
                with cv_col2:
                    st.metric("CV RMSE", f"{np.mean(self.cv_results['rmse']):.4f} ± {np.std(self.cv_results['rmse']):.4f}")
                with cv_col3:
                    st.metric("CV MAE", f"{np.mean(self.cv_results['mae']):.4f} ± {np.std(self.cv_results['mae']):.4f}")
                with cv_col4:
                    st.metric("CV Folds", len(self.cv_results['r2']))
            
            # Training progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            losses = []
            
            # Train model
            for epoch in range(epochs):
                loss = self.model.train_epoch(X, y)
                losses.append(loss)
                
                # Update progress
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
            
            # Final predictions and evaluation
            y_pred = self.model.forward_pass(X)
            evaluation_metrics = self.comprehensive_evaluation(y, y_pred)
            
            # Store results
            results = {
                'metrics': evaluation_metrics,
                'training_loss': losses,
                'feature_importance': self._compute_feature_importance(X, y),
                'feature_names': feature_names,
                'cv_results': self.cv_results,
                'model_architecture': {
                    'sequence_length': self.sequence_length,
                    'hidden_units': 64,
                    'learning_rate': 0.001,
                    'optimizer': 'Adam',
                    'activation_functions': ['ReLU', 'Tanh', 'Linear'],
                    'weight_initialization': ['He', 'Xavier']
                }
            }
            
            self.is_trained = True
            
            # Display comprehensive results
            self._display_training_results(results, y, y_pred)
            
            st.success("✅ Enhanced CNN-LSTM training completed successfully!")
            return results
            
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
            return None
    
    def _compute_feature_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute feature importance using permutation importance"""
        if self.model is None:
            return np.ones(X.shape[1])
        
        # Baseline performance
        baseline_pred = self.model.forward_pass(X)
        baseline_rmse = np.sqrt(mean_squared_error(y, baseline_pred))
        
        importance_scores = []
        
        for feature_idx in range(X.shape[1]):
            # Permute feature
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, feature_idx])
            
            # Compute performance with permuted feature
            permuted_pred = self.model.forward_pass(X_permuted)
            permuted_rmse = np.sqrt(mean_squared_error(y, permuted_pred))
            
            # Importance score is the increase in RMSE
            importance = permuted_rmse - baseline_rmse
            importance_scores.append(max(0, importance))  # Ensure non-negative
        
        # Normalize importance scores
        if np.sum(importance_scores) > 0:
            importance_scores = importance_scores / np.sum(importance_scores)
        
        return np.array(importance_scores)
    
    def _display_training_results(self, results: Dict, y_true: np.ndarray, y_pred: np.ndarray):
        """Display comprehensive training results"""
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Comprehensive Metrics")
            metrics = results['metrics']
            
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("R² Score", f"{metrics['r2']:.4f}")
                st.metric("MAE", f"{metrics['mae']:.4f}")
                st.metric("RMSE", f"{metrics['rmse']:.4f}")
            with metric_col2:
                st.metric("MAPE", f"{metrics['mape']:.2f}%")
                st.metric("Direction Accuracy", f"{metrics['direction_accuracy']:.2%}")
                st.metric("Explained Variance", f"{metrics['explained_variance']:.4f}")
            
            # Training loss curve
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                y=results['training_loss'],
                name='Training Loss',
                line=dict(color='red')
            ))
            fig_loss.update_layout(
                title='Training Loss Curve',
                xaxis_title='Epoch',
                yaxis_title='Loss (MSE)'
            )
            st.plotly_chart(fig_loss, use_container_width=True)
        
        with col2:
            # Actual vs Predicted
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=y_true, y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(color='blue')
            ))
            fig_pred.add_trace(go.Scatter(
                x=[y_true.min(), y_true.max()],
                y=[y_true.min(), y_true.max()],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            fig_pred.update_layout(
                title='Actual vs Predicted Values',
                xaxis_title='Actual NDVI',
                yaxis_title='Predicted NDVI'
            )
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Feature importance
            if 'feature_names' in results and len(results['feature_names']) > 0:
                top_features = min(10, len(results['feature_names']))
                fig_importance = px.bar(
                    x=results['feature_names'][:top_features],
                    y=results['feature_importance'][:top_features],
                    title='Top Feature Importance',
                    labels={'x': 'Features', 'y': 'Importance'}
                )
                st.plotly_chart(fig_importance, use_container_width=True)
        
        # Model architecture info
        with st.expander("🏗️ Model Architecture & Configuration"):
            st.json(results['model_architecture'])
            
            st.subheader("🎯 Optimization Details")
            st.info(f"**Optimizer:** Adam (β1=0.9, β2=0.999, ε=1e-8)")
            st.info(f"**Learning Rate:** {results['model_architecture']['learning_rate']}")
            st.info(f"**Weight Initialization:** {', '.join(results['model_architecture']['weight_initialization'])}")
            st.info(f"**Activation Functions:** {', '.join(results['model_architecture']['activation_functions'])}")
    
    def predict_future(self, data: List[Dict], years_ahead: int = 3) -> Optional[np.ndarray]:
        """Predict future values using trained model"""
        if not self.is_trained or self.model is None:
            st.error("❌ Model not trained. Please train the model first.")
            return None
        
        try:
            df = pd.DataFrame(data)
            ndvi_values = df['ndvi'].values
            
            if len(ndvi_values) < self.sequence_length:
                st.error(f"❌ Need at least {self.sequence_length} data points")
                return None
            
            predictions = []
            current_sequence = ndvi_values[-self.sequence_length:].copy()
            
            with st.spinner(f"🔮 CNN-LSTM predicting next {years_ahead} years..."):
                for i in range(years_ahead):
                    # Prepare features for current sequence
                    features = list(current_sequence)
                    features.extend([
                        np.mean(current_sequence), np.std(current_sequence),
                        np.min(current_sequence), np.max(current_sequence),
                        np.median(current_sequence), current_sequence[-1] - current_sequence[0]
                    ])
                    
                    # Add rolling features
                    for window in [2, 3]:
                        if len(current_sequence) >= window:
                            rolling_mean = np.mean(current_sequence[-window:])
                            features.append(rolling_mean)
                            if window > 1:
                                rolling_trend = current_sequence[-1] - current_sequence[-window]
                                features.append(rolling_trend)
                    
                    # Pad to consistent size
                    while len(features) < 20:
                        features.append(0.0)
                    if len(features) > 20:
                        features = features[:20]
                    
                    # Make prediction
                    features_array = np.array(features).reshape(1, -1)
                    prediction = self.model.forward_pass(features_array)[0]
                    
                    # Ensure reasonable bounds
                    prediction = max(0.1, min(0.9, prediction))
                    predictions.append(prediction)
                    
                    # Update sequence
                    current_sequence = np.roll(current_sequence, -1)
                    current_sequence[-1] = prediction
            
            st.success(f"✅ Generated {years_ahead} year predictions")
            return np.array(predictions)
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            return None
    
    def get_model_info(self) -> Dict:
        """Get comprehensive model information"""
        info = {
            'architecture': 'Enhanced CNN-LSTM',
            'sequence_length': self.sequence_length,
            'trained': self.is_trained,
            'features_used': len(self.feature_names) if self.feature_names else 0,
            'optimization': 'Adam with Cross-Validation',
            'evaluation_metrics': [
                'MSE', 'MAE', 'RMSE', 'R²', 'MAPE', 
                'Explained Variance', 'Direction Accuracy'
            ]
        }
        
        if self.cv_results:
            info['cv_performance'] = {
                'mean_r2': np.mean(self.cv_results['r2']),
                'mean_rmse': np.mean(self.cv_results['rmse']),
                'cv_folds': len(self.cv_results['r2'])
            }
        
        return info