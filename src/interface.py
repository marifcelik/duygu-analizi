import json
from typing import Dict, List
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Input, Static, TextArea, Pretty
from textual.binding import Binding
import asyncio


class EmotionAnalysisApp(App):
    """Textual app for emotion analysis"""
    
    CSS = """
    .title {
        dock: top;
        height: 3;
        background: $primary;
        color: $text;
        content-align: center middle;
        text-style: bold;
    }
    
    .input-container {
        dock: top;
        height: 8;
        background: $surface;
        padding: 1;
    }
    
    .results-container {
        background: $panel;
        padding: 1;
        margin: 1;
    }
    
    .emotion-result {
        background: $success;
        color: $text;
        margin: 1;
        padding: 1;
        border: solid $primary;
    }
    
    .confidence-bar {
        background: $warning;
        height: 1;
        margin: 1;
    }
    
    Button {
        margin: 1;
    }
    
    #analyze_button {
        background: $primary;
        color: $text;
        dock: bottom;
        height: 3;
        margin: 1;
    }
    
    #text_input {
        height: 5;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+r", "analyze", "Analyze"),
        Binding("ctrl+l", "clear", "Clear"),
    ]
    
    def __init__(self, predictor=None):
        super().__init__()
        self.predictor = predictor
        self.last_results = None
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        
        with Container():
            yield Static("🎭 Turkish Emotion Analysis", classes="title")
            
            with Vertical(classes="input-container"):
                yield Static("Enter your text below:")
                yield TextArea(
                    placeholder="Bugün çok mutluyum ve harika bir gün geçiriyorum!",
                    id="text_input"
                )
                yield Button("Analyze Emotion", id="analyze_button", variant="primary")
            
            yield Static("Results will appear here...", id="results", classes="results-container")
        
        yield Footer()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "analyze_button":
            self.analyze_text()
    
    def action_analyze(self) -> None:
        """Analyze text action."""
        self.analyze_text()
    
    def action_clear(self) -> None:
        """Clear text action."""
        text_input = self.query_one("#text_input", TextArea)
        text_input.text = ""
        results = self.query_one("#results", Static)
        results.update("Results will appear here...")
    
    def analyze_text(self) -> None:
        """Analyze the input text for emotions."""
        text_input = self.query_one("#text_input", TextArea)
        text = text_input.text.strip()
        
        if not text:
            self.show_error("Please enter some text to analyze.")
            return
        
        if self.predictor is None:
            self.show_error("No model loaded. Please train a model first.")
            return
        
        try:
            # Get prediction
            results = self.predictor.predict_emotion(text)
            self.display_results(results)
            
        except Exception as e:
            self.show_error(f"Error analyzing text: {str(e)}")
    
    def display_results(self, results: Dict) -> None:
        """Display emotion analysis results."""
        self.last_results = results
        
        # Main prediction
        main_emotion = results['predicted_emotion']
        top_emotions = results['top_emotions']
        
        # Create formatted result
        result_text = f"🎯 **Primary Emotion**: {main_emotion.upper()}\n\n"
        result_text += "📊 **Confidence Scores**:\n\n"
        
        for i, emotion_data in enumerate(top_emotions):
            emotion = emotion_data['emotion']
            confidence = emotion_data['confidence']
            bar_length = int(confidence * 20)  # Scale to 20 chars
            bar = "█" * bar_length + "░" * (20 - bar_length)
            result_text += f"{i+1}. {emotion.capitalize()}: {confidence:.3f} [{bar}]\n"
        
        # Add JSON output
        result_text += f"\n📄 **JSON Output**:\n```json\n{json.dumps(results, indent=2, ensure_ascii=False)}\n```"
        
        results_widget = self.query_one("#results", Static)
        results_widget.update(result_text)
    
    def show_error(self, message: str) -> None:
        """Show error message."""
        results = self.query_one("#results", Static)
        results.update(f"❌ **Error**: {message}")


class EmotionPredictor:
    """Wrapper class for making predictions with trained models"""
    
    def __init__(self, model, feature_extractor, label_encoder):
        self.model = model
        self.feature_extractor = feature_extractor
        self.label_encoder = label_encoder
    
    def predict_emotion(self, text: str) -> Dict:
        """Predict emotion for a given text"""
        # Extract features
        features = self.feature_extractor.extract_tfidf_features([text])
        
        # Get prediction with probabilities
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        # Get top 3 emotions
        top_indices = probabilities.argsort()[-3:][::-1]
        top_emotions = []
        
        for idx in top_indices:
            emotion = self.label_encoder.inverse_transform([idx])[0]
            confidence = probabilities[idx]
            top_emotions.append({
                'emotion': emotion,
                'confidence': float(confidence)
            })
        
        predicted_emotion = self.label_encoder.inverse_transform([prediction])[0]
        
        return {
            'text': text,
            'predicted_emotion': predicted_emotion,
            'top_emotions': top_emotions
        }


def run_interface(model=None, feature_extractor=None, label_encoder=None):
    """Run the Textual interface"""
    predictor = None
    if model and feature_extractor and label_encoder:
        predictor = EmotionPredictor(model, feature_extractor, label_encoder)
    
    app = EmotionAnalysisApp(predictor)
    app.run()


if __name__ == "__main__":
    # For testing purposes
    print("Running emotion analysis interface...")
    print("Note: No model loaded. Please train a model first.")
    run_interface() 