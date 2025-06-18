import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from typing import Dict, List, Any
import joblib


class EmotionClassifier:
    """Base class for emotion classification models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the model"""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        self.model = joblib.load(filepath)
        self.is_trained = True


class NaiveBayesClassifier(EmotionClassifier):
    """Naive Bayes classifier for emotion analysis"""
    
    def __init__(self, alpha: float = 1.0):
        super().__init__("Naive Bayes")
        self.alpha = alpha
        self.model = MultinomialNB(alpha=alpha)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train Naive Bayes model"""
        self.model.fit(X_train, y_train)
        self.is_trained = True


class LogisticRegressionClassifier(EmotionClassifier):
    """Logistic Regression classifier for emotion analysis"""
    
    def __init__(self, max_iter: int = 1000, C: float = 1.0, random_state: int = 42):
        super().__init__("Logistic Regression")
        self.max_iter = max_iter
        self.C = C
        self.random_state = random_state
        self.model = LogisticRegression(
            max_iter=max_iter, 
            C=C, 
            random_state=random_state,
            multi_class='ovr'
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train Logistic Regression model"""
        self.model.fit(X_train, y_train)
        self.is_trained = True


class RandomForestClassifier(EmotionClassifier):
    """Random Forest classifier for emotion analysis"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = None, random_state: int = 42):
        super().__init__("Random Forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train Random Forest model"""
        self.model.fit(X_train, y_train)
        self.is_trained = True


class SimpleNeuralNetwork(EmotionClassifier):
    """Simple neural network using PyTorch for emotion classification"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 6, 
                 learning_rate: float = 0.001, epochs: int = 100):
        super().__init__("Simple Neural Network")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = 'cpu'  # No GPU available as per user requirements
        
        # Import torch here to avoid dependency issues if not needed
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            self.torch = torch
            self.nn = nn
            self.optim = optim
            self._build_model()
        except ImportError:
            print("PyTorch not available. Falling back to simpler models.")
            self.model = None
    
    def _build_model(self):
        """Build the neural network architecture"""
        class SimpleNN(self.nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super(SimpleNN, self).__init__()
                self.fc1 = self.nn.Linear(input_dim, hidden_dim)
                self.fc2 = self.nn.Linear(hidden_dim, hidden_dim // 2)
                self.fc3 = self.nn.Linear(hidden_dim // 2, output_dim)
                self.relu = self.nn.ReLU()
                self.dropout = self.nn.Dropout(0.3)
                
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x
        
        self.model = SimpleNN(self.input_dim, self.hidden_dim, self.output_dim)
        self.criterion = self.nn.CrossEntropyLoss()
        self.optimizer = self.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the neural network"""
        if self.model is None:
            raise ValueError("PyTorch not available")
        
        # Convert to tensors
        X_tensor = self.torch.FloatTensor(X_train)
        y_tensor = self.torch.LongTensor(y_train)
        
        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("PyTorch not available")
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        with self.torch.no_grad():
            X_tensor = self.torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            _, predicted = self.torch.max(outputs.data, 1)
        
        return predicted.numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("PyTorch not available")
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        with self.torch.no_grad():
            X_tensor = self.torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            probabilities = self.torch.softmax(outputs, dim=1)
        
        return probabilities.numpy()


class ModelEvaluator:
    """Evaluate and compare different models"""
    
    @staticmethod
    def evaluate_model(model: EmotionClassifier, X_test: np.ndarray, y_test: np.ndarray, 
                      label_names: List[str] = None) -> Dict[str, Any]:
        """Evaluate a single model"""
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        # Basic metrics
        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, predictions, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_test, predictions, average=None
        )
        
        # Classification report
        target_names = label_names if label_names else [f"Class_{i}" for i in range(len(np.unique(y_test)))]
        class_report = classification_report(y_test, predictions, target_names=target_names, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, predictions)
        
        return {
            'model_name': model.model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    @staticmethod
    def compare_models(models: List[EmotionClassifier], X_test: np.ndarray, y_test: np.ndarray,
                      label_names: List[str] = None) -> pd.DataFrame:
        """Compare multiple models"""
        results = []
        
        for model in models:
            if model.is_trained:
                eval_results = ModelEvaluator.evaluate_model(model, X_test, y_test, label_names)
                results.append({
                    'Model': eval_results['model_name'],
                    'Accuracy': eval_results['accuracy'],
                    'Precision': eval_results['precision'],
                    'Recall': eval_results['recall'],
                    'F1-Score': eval_results['f1_score']
                })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def get_prediction_with_confidence(model: EmotionClassifier, X: np.ndarray, 
                                     label_encoder, top_k: int = 3) -> List[Dict]:
        """Get predictions with confidence scores"""
        probabilities = model.predict_proba(X)
        predictions = model.predict(X)
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            # Get top k predictions
            top_indices = np.argsort(probs)[-top_k:][::-1]
            top_emotions = []
            
            for idx in top_indices:
                emotion = label_encoder.inverse_transform([idx])[0]
                confidence = probs[idx]
                top_emotions.append({
                    'emotion': emotion,
                    'confidence': float(confidence)
                })
            
            results.append({
                'predicted_emotion': label_encoder.inverse_transform([pred])[0],
                'top_emotions': top_emotions
            })
        
        return results 