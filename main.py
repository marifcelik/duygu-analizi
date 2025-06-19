#!/usr/bin/env python3

import os
import argparse
import numpy as np
import joblib
import time

from src.data_processor import DataProcessor
from src.feature_extraction import FeatureExtractor
from src.models import (
    NaiveBayesClassifier, 
    LogisticRegressionClassifier, 
    SimpleNeuralNetwork,
    ModelEvaluator
)
from src.interface import run_interface

class TrainingProgress:
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
    
    def update(self, step_name, add_step=True):
        if add_step:
            self.current_step += 1
        
        elapsed = time.time() - self.start_time
        progress = (self.current_step / self.total_steps) * 100
        
        bar_length = 30
        filled = int(bar_length * self.current_step / self.total_steps)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\r[{bar}] {progress:.1f}% | Step {self.current_step}/{self.total_steps}: {step_name} | {elapsed:.1f}s", end='', flush=True)
        
        if self.current_step == self.total_steps:
            print()
    
    def complete_step(self, step_name):
        """Mark a step as complete"""
        self.update(step_name, add_step=True)
        print()


def train_models(data_path: str = "data", save_models: bool = True):
    print("Starting Emotion Analysis Training...")
    
    # Initialize progress tracker (7 main steps)
    progress = TrainingProgress(7)
    
    # Step 1: Load and preprocess data
    progress.complete_step("Loading and preprocessing data")
    processor = DataProcessor(data_path)
    df = processor.load_data()
    df = processor.additional_preprocessing(df)
    
    print(f"Dataset size: {len(df)} samples")
    print(f"Emotion distribution:")
    for emotion, count in processor.get_emotion_distribution(df).items():
        print(f"  {emotion}: {count}")
    
    # Step 2: Split data
    progress.complete_step("Splitting data")
    train_df, test_df = processor.train_test_split(df, test_size=0.2, random_state=42)
    print(f"Training set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Step 3: Extract features
    progress.complete_step("Extracting features")
    feature_extractor = FeatureExtractor()
    
    X_train_tfidf = feature_extractor.extract_tfidf_features(train_df['text'].tolist())
    X_test_tfidf = feature_extractor.extract_tfidf_features(test_df['text'].tolist())
    
    X_train_stats = feature_extractor.extract_statistical_features(train_df['text'].tolist())
    X_test_stats = feature_extractor.extract_statistical_features(test_df['text'].tolist())
    
    X_train = feature_extractor.combine_features(X_train_tfidf, X_train_stats)
    X_test = feature_extractor.combine_features(X_test_tfidf, X_test_stats)
    
    y_train = feature_extractor.encode_labels(train_df['label'].tolist())
    y_test = feature_extractor.encode_labels(test_df['label'].tolist())
    
    print(f"Feature dimensions: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    progress.complete_step("Training models")
    models = []
    
    print("  Training Naive Bayes...")
    nb_model = NaiveBayesClassifier(alpha=0.1)
    nb_model.train(X_train, y_train)
    models.append(nb_model)
    
    print("  Training Logistic Regression...")
    lr_model = LogisticRegressionClassifier(max_iter=1000, C=1.0)
    lr_model.train(X_train, y_train)
    models.append(lr_model)
    
    try:
        print("  Training Simple Neural Network...")
        nn_model = SimpleNeuralNetwork(
            input_dim=X_train.shape[1],
            hidden_dim=128,
            output_dim=len(np.unique(y_train)),
            epochs=50
        )
        nn_model.train(X_train, y_train)
        models.append(nn_model)
    except Exception as e:
        print(f"  Neural Network training failed: {e}")
    
    progress.complete_step("Evaluating models")
    label_names = feature_extractor.label_encoder.classes_
    
    evaluation_results = []
    best_model = None
    best_f1 = 0
    
    for model in models:
        print(f"  Evaluating {model.model_name}...")
        results = ModelEvaluator.evaluate_model(model, X_test, y_test, label_names)
        evaluation_results.append(results)
        
        print(f"    Accuracy: {results['accuracy']:.4f}")
        print(f"    Precision: {results['precision']:.4f}")
        print(f"    Recall: {results['recall']:.4f}")
        print(f"    F1-Score: {results['f1_score']:.4f}")
        
        if results['f1_score'] > best_f1:
            best_f1 = results['f1_score']
            best_model = model
    
    # Step 6: Compare models
    progress.complete_step("Comparing models")
    comparison_df = ModelEvaluator.compare_models(models, X_test, y_test, label_names)
    print(comparison_df.to_string(index=False))
    
    # Step 7: Save models
    if save_models and best_model:
        progress.complete_step("Saving models")
        os.makedirs("models", exist_ok=True)
        
        best_model.save_model("models/best_model.joblib")
        joblib.dump(feature_extractor, "models/feature_extractor.joblib")
        
        print(f"✅ Training completed! Best model: {best_model.model_name}")
        print(f"🎯 Best F1-Score: {best_f1:.4f}")
        print(f"⏱️ Total time: {time.time() - progress.start_time:.1f}s")
    else:
        progress.complete_step("Training completed")
        print(f"✅ Training completed!")
        print(f"⏱️ Total time: {time.time() - progress.start_time:.1f}s")
    
    return best_model, feature_extractor, evaluation_results


def load_trained_model():
    """Load pre-trained model for inference"""
    try:
        from sklearn.base import BaseEstimator
        import joblib
        
        feature_extractor = joblib.load("models/feature_extractor.joblib")
        
        class SavedModel:
            def __init__(self, model_path):
                self.model = joblib.load(model_path)
                self.is_trained = True
                self.model_name = "Loaded Model"
            
            def predict(self, X):
                return self.model.predict(X)
            
            def predict_proba(self, X):
                return self.model.predict_proba(X)
        
        model = SavedModel("models/best_model.joblib")
        
        return model, feature_extractor
    
    except FileNotFoundError:
        print("❌ No trained model found. Please run training first.")
        return None, None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None


def run_inference_mode():
    """Run the application in inference mode"""
    print("🔮 Loading trained model...")
    model, feature_extractor = load_trained_model()
    
    if model is None:
        print("Please train a model first using: python main.py --train")
        return
    
    print("✅ Model loaded successfully!")
    print("🎭 Starting Emotion Analysis Interface...")
    
    label_encoder = feature_extractor.label_encoder
    
    run_interface(model, feature_extractor, label_encoder)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Turkish Emotion Analysis")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--interface", action="store_true", help="Run interface")
    parser.add_argument("--data-path", default="data", help="Path to data directory")
    parser.add_argument("--no-save", action="store_true", help="Don't save trained models")
    
    args = parser.parse_args()
    
    if args.train:
        train_models(args.data_path, save_models=not args.no_save)
    elif args.interface:
        run_inference_mode()
    else:
        if not os.path.exists("models/best_model.joblib"):
            print("No trained models found. Starting training...")
            train_models(args.data_path, save_models=True)
        
        print("\n" + "="*50)
        print("🎭 TURKISH EMOTION ANALYSIS")
        print("="*50)
        run_inference_mode()


if __name__ == "__main__":
    main()
