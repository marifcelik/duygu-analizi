import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List, Any
import re


class FeatureExtractor:
    """Extract features from text for emotion classification"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
    
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-ZğüşöçıĞÜŞÖÇİ\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_tfidf_features(self, texts: List[str], max_features: int = 5000, 
                              ngram_range: Tuple[int, int] = (1, 2)) -> np.ndarray:
        """Extract TF-IDF features"""
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words=None,  # We'll handle Turkish stop words separately if needed
                lowercase=False,  # Already handled in preprocessing
                strip_accents=None
            )
            features = self.tfidf_vectorizer.fit_transform(processed_texts)
        else:
            features = self.tfidf_vectorizer.transform(processed_texts)
        
        return features.toarray()
    
    def extract_bow_features(self, texts: List[str], max_features: int = 5000,
                            ngram_range: Tuple[int, int] = (1, 1)) -> np.ndarray:
        """Extract Bag-of-Words features"""
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        if self.count_vectorizer is None:
            self.count_vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words=None,
                lowercase=False,
                strip_accents=None
            )
            features = self.count_vectorizer.fit_transform(processed_texts)
        else:
            features = self.count_vectorizer.transform(processed_texts)
        
        return features.toarray()
    
    def extract_statistical_features(self, texts: List[str]) -> np.ndarray:
        """Extract statistical features from text"""
        features = []
        
        for text in texts:
            if pd.isna(text):
                text = ""
            text = str(text)
            
            # Text length features
            char_count = len(text)
            word_count = len(text.split())
            sentence_count = len([s for s in text.split('.') if s.strip()])
            
            # Average word length
            avg_word_length = np.mean([len(word) for word in text.split()]) if word_count > 0 else 0
            
            # Punctuation features
            exclamation_count = text.count('!')
            question_count = text.count('?')
            comma_count = text.count(',')
            
            # Capital letters ratio
            capital_ratio = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
            
            features.append([
                char_count, word_count, sentence_count, avg_word_length,
                exclamation_count, question_count, comma_count, capital_ratio
            ])
        
        return np.array(features)
    
    def encode_labels(self, labels: List[str]) -> np.ndarray:
        """Encode string labels to numeric"""
        return self.label_encoder.fit_transform(labels)
    
    def decode_labels(self, encoded_labels: np.ndarray) -> List[str]:
        """Decode numeric labels back to strings"""
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def get_feature_names(self, feature_type: str = 'tfidf') -> List[str]:
        """Get feature names for interpretability"""
        if feature_type == 'tfidf' and self.tfidf_vectorizer is not None:
            return list(self.tfidf_vectorizer.get_feature_names_out())
        elif feature_type == 'bow' and self.count_vectorizer is not None:
            return list(self.count_vectorizer.get_feature_names_out())
        elif feature_type == 'statistical':
            return [
                'char_count', 'word_count', 'sentence_count', 'avg_word_length',
                'exclamation_count', 'question_count', 'comma_count', 'capital_ratio'
            ]
        else:
            return []
    
    def combine_features(self, *feature_arrays) -> np.ndarray:
        """Combine multiple feature arrays"""
        return np.hstack(feature_arrays)
    
    def get_top_features_per_class(self, features: np.ndarray, labels: np.ndarray, 
                                   feature_names: List[str], top_k: int = 10) -> dict:
        """Get top features for each emotion class"""
        if len(feature_names) != features.shape[1]:
            return {}
        
        unique_labels = np.unique(labels)
        top_features = {}
        
        for label in unique_labels:
            label_mask = labels == label
            label_features = features[label_mask]
            
            # Calculate mean feature values for this label
            mean_features = np.mean(label_features, axis=0)
            
            # Get top features
            top_indices = np.argsort(mean_features)[-top_k:][::-1]
            top_features[label] = [
                (feature_names[i], mean_features[i]) 
                for i in top_indices
            ]
        
        return top_features 