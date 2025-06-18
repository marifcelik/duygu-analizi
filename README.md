# 🎭 Turkish Emotion Analysis

A comprehensive machine learning project for classifying Turkish text by emotional content using classical ML techniques and simple neural networks.

## 🎯 Project Objective

Develop a model that classifies sentences by determining which emotion they reflect (joy, sadness, anger, fear, surprise, disgust). Given a sentence, the model will predict the related emotion and display the result to the user through an interactive terminal interface.

## ✨ Features

- **Multiple ML Models**: Naive Bayes, Logistic Regression, and Simple Neural Networks
- **Feature Engineering**: TF-IDF, Bag-of-Words, and statistical text features
- **Interactive Interface**: Beautiful terminal interface using Textual
- **Model Comparison**: Comprehensive evaluation with accuracy, precision, recall, and F1-score
- **Turkish Language Support**: Optimized for Turkish text analysis
- **JSON Output**: Machine-readable prediction results

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- UV package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd duygu-analizi
```

2. Install dependencies using UV:
```bash
uv sync
```

3. Run the application:
```bash
# Train models and start the interface
python main.py

# Or train models only
python main.py --train

# Or run interface only (requires pre-trained models)
python main.py --interface
```

## 📊 Dataset

The project uses the TREMODATA.xml dataset for Turkish emotion analysis. The dataset contains labeled sentences with emotions:

- **joy** (mutlu/Happy)
- **sadness** (üzgün/Sadness)
- **anger** (kızgın/Anger)
- **fear** (korku/Fear)
- **surprise** (surpriz/Surprise)
- **disgust** (Disgust)

## 🧠 Models

### Classical ML Models
- **Naive Bayes**: Fast and effective for text classification
- **Logistic Regression**: Linear model with good interpretability
- **Random Forest**: Ensemble method for robust predictions

### Deep Learning
- **Simple Neural Network**: 3-layer feedforward network using PyTorch
- CPU-optimized (no GPU required)

### Feature Engineering
- **TF-IDF**: Term frequency-inverse document frequency vectors
- **Bag of Words**: Simple word occurrence counting
- **Statistical Features**: Text length, punctuation, capitalization patterns

## 🖥️ Interface

The application features a modern terminal interface built with Textual:

- **Real-time Analysis**: Type text and get instant emotion predictions
- **Confidence Scores**: Visual confidence bars for all emotions
- **JSON Output**: Machine-readable results for integration
- **Keyboard Shortcuts**: 
  - `Ctrl+R`: Analyze text
  - `Ctrl+L`: Clear input
  - `Ctrl+C`: Quit

## 📈 Evaluation Metrics

The system provides comprehensive evaluation:
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class and weighted precision
- **Recall**: Per-class and weighted recall
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed error analysis

## 🗂️ Project Structure

```
duygu-analizi/
├── src/
│   ├── __init__.py
│   ├── data_processor.py      # Data loading and preprocessing
│   ├── feature_extraction.py  # Feature engineering
│   ├── models.py             # ML models implementation
│   └── interface.py          # Textual UI
├── data/
│   └── TREMODATA.xml         # Dataset
├── models/                   # Saved models (created after training)
├── main.py                   # Main application entry point
├── pyproject.toml           # Dependencies and project config
└── README.md               # This file
```

## 🔧 Usage Examples

### Training Models

```bash
# Train all models and compare performance
python main.py --train

# Train with custom data path
python main.py --train --data-path /path/to/data

# Train without saving models
python main.py --train --no-save
```

### Using the Interface

```python
from src.interface import run_interface
from src.models import LogisticRegressionClassifier
from src.feature_extraction import FeatureExtractor

# Load your trained model
model = LogisticRegressionClassifier()
feature_extractor = FeatureExtractor()

# Run the interface
run_interface(model, feature_extractor, label_encoder)
```

### Programmatic Usage

```python
from src.data_processor import DataProcessor
from src.feature_extraction import FeatureExtractor
from src.models import NaiveBayesClassifier

# Load and process data
processor = DataProcessor()
df = processor.load_data()

# Extract features
extractor = FeatureExtractor()
features = extractor.extract_tfidf_features(df['text'].tolist())
labels = extractor.encode_labels(df['label'].tolist())

# Train model
model = NaiveBayesClassifier()
model.train(features, labels)

# Make predictions
predictions = model.predict(features)
```

## 🎨 Sample Predictions

```json
{
  "text": "Bugün çok mutluyum ve harika bir gün geçiriyorum!",
  "predicted_emotion": "joy",
  "top_emotions": [
    {"emotion": "joy", "confidence": 0.854},
    {"emotion": "surprise", "confidence": 0.089},
    {"emotion": "sadness", "confidence": 0.057}
  ]
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TREMO dataset contributors
- Turkish NLP community
- Textual framework developers

## 📞 Support

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ for Turkish NLP** 