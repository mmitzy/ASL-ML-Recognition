import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import time
import os
from collections import deque, Counter
import matplotlib.pyplot as plt

# Import existing model classes
from cnn import CNNModel
from nn import SimpleNeuralNetwork
from logisticregression import LogisticRegressionModel
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
# Note: classicalmodels.py doesn't export a main class, so we'll create a wrapper

class ClassicalModelsWrapper:
    """Wrapper for classical models (KNN, SVM, Random Forest)"""
    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.models = {}
        self.pca = None
        self.scaler = None
        self.device = 'cpu'
        
    def load_models(self):
        """Load classical models from pickle files"""
        try:
            # Load PCA transformer
            pca_path = os.path.join(self.models_dir, 'pca_transformer.pkl')
            if os.path.exists(pca_path):
                self.pca = joblib.load(pca_path)
            
            # Load Random Forest (usually best performing)
            rf_path = os.path.join(self.models_dir, 'rf_model.pkl')
            if os.path.exists(rf_path):
                self.models['rf'] = joblib.load(rf_path)
                
            # Load SVM
            svm_path = os.path.join(self.models_dir, 'svm_model.pkl')
            if os.path.exists(svm_path):
                self.models['svm'] = joblib.load(svm_path)
                
            # Load KNN
            knn_path = os.path.join(self.models_dir, 'knn_model.pkl')
            if os.path.exists(knn_path):
                self.models['knn'] = joblib.load(knn_path)
                
            # Create a simple scaler for preprocessing
            self.scaler = StandardScaler()
            return len(self.models) > 0
        except Exception as e:
            print(f"Error loading classical models: {e}")
            return False
    
    def to(self, device):
        """Compatibility method"""
        return self
        
    def eval(self):
        """Compatibility method"""
        pass
        
    def predict_proba(self, image_tensor):
        """Make predictions using classical models"""
        if not self.models or self.pca is None:
            return torch.zeros(29)
            
        # Convert tensor to numpy and flatten
        image_np = image_tensor.cpu().numpy().flatten().reshape(1, -1)
        
        # Use Random Forest as primary model (usually best performer)
        if 'rf' in self.models:
            try:
                # Simple preprocessing - normalize pixel values
                image_np = image_np / 255.0  # Normalize to [0,1]
                
                # For demonstration, we'll use a simplified approach
                # In practice, you'd want to use the same PCA preprocessing as in training
                
                # Use only a subset of features to avoid dimensionality issues
                if image_np.shape[1] > 2000:
                    # Take every nth pixel to reduce dimensions
                    step = image_np.shape[1] // 2000
                    image_np = image_np[:, ::step]
                
                # Get prediction probabilities
                if hasattr(self.models['rf'], 'predict_proba'):
                    probs = self.models['rf'].predict_proba(image_np)[0]
                    # Pad or truncate to 29 classes
                    if len(probs) < 29:
                        padded_probs = np.zeros(29)
                        padded_probs[:len(probs)] = probs
                        probs = padded_probs
                    elif len(probs) > 29:
                        probs = probs[:29]
                    
                    return torch.tensor(probs, dtype=torch.float32)
                else:
                    # Fallback for models without predict_proba
                    pred = self.models['rf'].predict(image_np)[0]
                    probs = torch.zeros(29)
                    if pred < 29:
                        probs[pred] = 1.0
                    return probs
                    
            except Exception as e:
                print(f"Error in classical model prediction: {e}")
                
        return torch.zeros(29)

class ASLSentenceDemo:
    """
    Demo class for real-time ASL sentence recognition using ensemble learning.
    Combines multiple models to predict complete words from streams of ASL images.
    """
    
    def __init__(self, device='cpu', confidence_threshold=0.7, stability_frames=8):
        """
        Initialize the ASL Sentence Demo
        
        Args:
            device: Device to run models on ('cpu' or 'cuda')
            confidence_threshold: Minimum confidence for predictions
            stability_frames: Number of consistent frames needed for word recognition
        """
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.stability_frames = stability_frames
        
        # Class names mapping (ASL alphabet + special characters)
        # Order matches PyTorch ImageFolder alphabetical sorting
        self.class_names = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
            'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 
            'Y', 'Z', 'del', 'nothing', 'space'
        ]
        
        # Initialize models
        self.models = {}
        self.model_weights = {}
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        # Sentence construction
        self.current_word = ""
        self.completed_words = []
        self.letter_buffer = deque(maxlen=stability_frames)
        self.last_prediction = None
        self.last_change_time = time.time()
        
        # Statistics
        self.prediction_history = []
        self.confidence_history = []
        
        print("🚀 ASL Sentence Demo Initialized")
        print(f"📱 Device: {self.device}")
        print(f"🎯 Confidence Threshold: {confidence_threshold}")
        print(f"📊 Stability Frames: {stability_frames}")
        print("="*60)
    
    def load_models(self, model_paths):
        """
        Load and initialize all available models
        
        Args:
            model_paths: Dictionary with model names and their file paths
        """
        print("📂 Loading Models...")
        
        # Load CNN Model
        if 'cnn' in model_paths and os.path.exists(model_paths['cnn']):
            try:
                self.models['cnn'] = CNNModel(num_classes=29)
                self.models['cnn'].load_model(model_paths['cnn'], self.device)
                self.models['cnn'].eval()
                self.model_weights['cnn'] = 0.9  # 90% weight for CNN
                print("✅ CNN Model loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load CNN: {e}")
        
        # Load Neural Network Model
        if 'nn' in model_paths and os.path.exists(model_paths['nn']):
            try:
                self.models['nn'] = SimpleNeuralNetwork(
                    input_dim=64*64*3, 
                    hidden_dim=512, 
                    num_classes=29
                )
                self.models['nn'].load_model(model_paths['nn'], self.device)
                self.models['nn'].eval()
                self.model_weights['nn'] = 0.1  # 10% weight for FFNN
                print("✅ Neural Network Model loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load NN: {e}")
        
        # Load Classical Models - DISABLED for now
        # if 'classical' in model_paths and os.path.exists(model_paths['classical']):
        #     try:
        #         self.models['classical'] = ClassicalModelsWrapper(model_paths['classical'])
        #         if self.models['classical'].load_models():
        #             self.model_weights['classical'] = 0.3  # 30% weight for classical
        #             print("✅ Classical Models loaded successfully")
        #         else:
        #             print("⚠️ No classical models found")
        #             del self.models['classical']
        #     except Exception as e:
        #         print(f"❌ Failed to load Classical Models: {e}")
        
        # Skip Logistic Regression (as requested)
        
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            self.model_weights = {k: v/total_weight for k, v in self.model_weights.items()}
        
        print(f"🎯 Loaded {len(self.models)} models with weights: {self.model_weights}")
        print("="*60)
    
    def ensemble_predict(self, image_tensor):
        """
        Make ensemble predictions using all loaded models
        
        Args:
            image_tensor: Preprocessed image tensor
            
        Returns:
            Tuple of (predicted_class, confidence, individual_predictions)
        """
        if not self.models:
            raise ValueError("No models loaded!")
        
        ensemble_probs = torch.zeros(29).to(self.device)
        individual_predictions = {}
        
        with torch.no_grad():
            for model_name, model in self.models.items():
                try:
                    if model_name in ['cnn', 'nn']:
                        # Deep learning models
                        outputs = model(image_tensor.unsqueeze(0))
                        probs = F.softmax(outputs, dim=1).squeeze()
                    elif model_name == 'classical':
                        # Classical models
                        probs = model.predict_proba(image_tensor)
                    else:
                        # Skip unsupported model types
                        continue
                    
                    weight = self.model_weights[model_name]
                    ensemble_probs += weight * probs
                    
                    # Store individual prediction
                    individual_predictions[model_name] = {
                        'class': torch.argmax(probs).item(),
                        'confidence': torch.max(probs).item(),
                        'probs': probs.cpu().numpy()
                    }
                    
                except Exception as e:
                    print(f"⚠️ Error in {model_name}: {e}")
                    continue
        
        # Final ensemble prediction
        predicted_class = torch.argmax(ensemble_probs).item()
        confidence = torch.max(ensemble_probs).item()
        
        return predicted_class, confidence, individual_predictions
    
    def process_image(self, image):
        """
        Process a single image and update sentence construction
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Dictionary with prediction results
        """
        # Convert to PIL if numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Preprocess image
        image_tensor = self.transform(image).to(self.device)
        
        # Make ensemble prediction
        predicted_class, confidence, individual_preds = self.ensemble_predict(image_tensor)
        predicted_letter = self.class_names[predicted_class]
        
        # Store prediction history
        self.prediction_history.append(predicted_letter)
        self.confidence_history.append(confidence)
        
        # Only process if confidence is above threshold
        if confidence >= self.confidence_threshold:
            self.letter_buffer.append(predicted_letter)
            
            # Check for stable prediction
            if len(self.letter_buffer) == self.stability_frames:
                most_common = Counter(self.letter_buffer).most_common(1)[0]
                stable_letter, count = most_common
                
                # If majority of frames agree and it's different from last prediction
                if count >= self.stability_frames * 0.6 and stable_letter != self.last_prediction:
                    self._handle_stable_prediction(stable_letter)
                    self.last_prediction = stable_letter
                    self.last_change_time = time.time()
        
        return {
            'predicted_letter': predicted_letter,
            'confidence': confidence,
            'current_word': self.current_word,
            'completed_words': self.completed_words.copy(),
            'sentence': ' '.join(self.completed_words) + (' ' + self.current_word if self.current_word else ''),
            'individual_predictions': individual_preds,
            'stable': len(self.letter_buffer) == self.stability_frames and confidence >= self.confidence_threshold
        }
    
    def _handle_stable_prediction(self, letter):
        """Handle a stable letter prediction"""
        if letter == 'space':
            if self.current_word:
                self.completed_words.append(self.current_word)
                self.current_word = ""
                print(f"🔤 Word completed: '{self.completed_words[-1]}'")
        elif letter == 'del':
            if self.current_word:
                self.current_word = self.current_word[:-1]
                print(f"⌫ Letter deleted. Current word: '{self.current_word}'")
        elif letter != 'nothing':
            self.current_word += letter
            print(f"📝 Letter added: '{letter}' → Current word: '{self.current_word}'")
    
    def process_video_stream(self, video_source=0, duration=60):
        """
        Process live video stream for real-time ASL recognition
        
        Args:
            video_source: Video source (0 for webcam, or path to video file)
            duration: Duration in seconds to run the demo
        """
        print(f"🎥 Starting video stream demo for {duration} seconds...")
        print("Controls:")
        print("  - 'q': Quit")
        print("  - 'r': Reset current sentence")
        print("  - 's': Save current sentence")
        print("="*60)
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {video_source}")
        
        start_time = time.time()
        frame_count = 0
        fps_counter = deque(maxlen=30)
        
        try:
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_start = time.time()
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                # Process frame
                result = self.process_image(pil_image)
                
                # Calculate FPS
                frame_time = time.time() - frame_start
                fps_counter.append(1.0 / frame_time if frame_time > 0 else 0)
                avg_fps = np.mean(fps_counter)
                
                # Draw results on frame
                self._draw_results_on_frame(frame, result, avg_fps)
                
                # Display frame
                cv2.imshow('ASL Sentence Recognition Demo', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.reset_sentence()
                    print("🔄 Sentence reset")
                elif key == ord('s'):
                    self.save_sentence()
                
                frame_count += 1
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        print(f"\n🎬 Demo completed! Processed {frame_count} frames")
        self.print_final_statistics()
    
    def _draw_results_on_frame(self, frame, result, fps):
        """Draw prediction results on the video frame"""
        height, width = frame.shape[:2]
        
        # Create overlay
        overlay = frame.copy()
        
        # Background for text
        cv2.rectangle(overlay, (10, 10), (width-10, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Text information
        y_offset = 40
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Current prediction
        pred_text = f"Prediction: {result['predicted_letter']} ({result['confidence']:.2f})"
        color = (0, 255, 0) if result['stable'] else (0, 255, 255)
        cv2.putText(frame, pred_text, (20, y_offset), font, 0.7, color, 2)
        y_offset += 30
        
        # Current word
        word_text = f"Current Word: {result['current_word']}"
        cv2.putText(frame, word_text, (20, y_offset), font, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        # Sentence
        sentence_text = f"Sentence: {result['sentence']}"
        if len(sentence_text) > 50:
            sentence_text = sentence_text[:47] + "..."
        cv2.putText(frame, sentence_text, (20, y_offset), font, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        # FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (20, y_offset), font, 0.5, (255, 255, 255), 1)
        
        # Stability indicator
        if result['stable']:
            cv2.circle(frame, (width-50, 50), 20, (0, 255, 0), -1)
            cv2.putText(frame, "STABLE", (width-120, 60), font, 0.5, (0, 255, 0), 2)
        else:
            cv2.circle(frame, (width-50, 50), 20, (0, 0, 255), -1)
            cv2.putText(frame, "UNSTABLE", (width-140, 60), font, 0.5, (0, 0, 255), 2)
    
    def process_image_sequence(self, image_paths):
        """
        Process a sequence of images to form words/sentences
        
        Args:
            image_paths: List of image file paths
        """
        print(f"📸 Processing {len(image_paths)} images...")
        
        results = []
        for i, image_path in enumerate(image_paths):
            if not os.path.exists(image_path):
                print(f"⚠️ Image not found: {image_path}")
                continue
            
            try:
                image = Image.open(image_path)
                result = self.process_image(image)
                results.append(result)
                
                print(f"[{i+1:3d}/{len(image_paths)}] {result['predicted_letter']} "
                      f"({result['confidence']:.3f}) → '{result['sentence']}'")
                
            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
        
        return results
    
    def reset_sentence(self):
        """Reset the current sentence and word"""
        self.current_word = ""
        self.completed_words = []
        self.letter_buffer.clear()
        self.last_prediction = None
    
    def save_sentence(self):
        """Save the current sentence to a file"""
        sentence = ' '.join(self.completed_words)
        if self.current_word:
            sentence += ' ' + self.current_word
        
        if sentence.strip():
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"asl_sentence_{timestamp}.txt"
            
            with open(filename, 'w') as f:
                f.write(sentence.strip())
            
            print(f"💾 Sentence saved to: {filename}")
            print(f"📝 Content: '{sentence.strip()}'")
        else:
            print("⚠️ No sentence to save!")
    
    def print_final_statistics(self):
        """Print final statistics about the session"""
        print("\n" + "="*60)
        print("📊 SESSION STATISTICS")
        print("="*60)
        
        if self.prediction_history:
            print(f"🎯 Total Predictions: {len(self.prediction_history)}")
            print(f"📈 Average Confidence: {np.mean(self.confidence_history):.3f}")
            print(f"🔤 Unique Letters: {len(set(self.prediction_history))}")
            
            # Most common predictions
            letter_counts = Counter(self.prediction_history)
            print(f"🏆 Most Common Letters: {letter_counts.most_common(5)}")
        
        # Final sentence
        final_sentence = ' '.join(self.completed_words)
        if self.current_word:
            final_sentence += ' ' + self.current_word
        
        if final_sentence.strip():
            print(f"📝 Final Sentence: '{final_sentence.strip()}'")
            print(f"🔤 Word Count: {len(self.completed_words)}")
        
        print("="*60)
    
    def plot_session_analytics(self):
        """Plot analytics from the current session"""
        if not self.prediction_history:
            print("⚠️ No data to plot!")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Confidence over time
        ax1.plot(self.confidence_history)
        ax1.axhline(y=self.confidence_threshold, color='r', linestyle='--', 
                   label=f'Threshold ({self.confidence_threshold})')
        ax1.set_title('Confidence Over Time')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Confidence')
        ax1.legend()
        ax1.grid(True)
        
        # Letter frequency
        letter_counts = Counter(self.prediction_history)
        letters, counts = zip(*letter_counts.most_common(15))
        ax2.bar(letters, counts)
        ax2.set_title('Letter Frequency (Top 15)')
        ax2.set_xlabel('Letters')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
        # Confidence distribution
        ax3.hist(self.confidence_history, bins=20, alpha=0.7, edgecolor='black')
        ax3.axvline(x=self.confidence_threshold, color='r', linestyle='--', 
                   label=f'Threshold ({self.confidence_threshold})')
        ax3.set_title('Confidence Distribution')
        ax3.set_xlabel('Confidence')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        # High confidence predictions over time
        high_conf_indices = [i for i, conf in enumerate(self.confidence_history) 
                           if conf >= self.confidence_threshold]
        high_conf_values = [self.confidence_history[i] for i in high_conf_indices]
        
        ax4.scatter(high_conf_indices, high_conf_values, alpha=0.6)
        ax4.set_title('High Confidence Predictions')
        ax4.set_xlabel('Frame')
        ax4.set_ylabel('Confidence')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('asl_session_analytics.png', dpi=300, bbox_inches='tight')
        print("📊 Session analytics saved as 'asl_session_analytics.png'")
        plt.show()


def main():
    """Main demo function"""
    print("🌟 ASL Sentence Recognition Demo")
    print("="*60)
    
    # Initialize demo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    demo = ASLSentenceDemo(
        device=device,
        confidence_threshold=0.7,
        stability_frames=8
    )
    
    # Define model paths
    model_paths = {
        'cnn': 'training_files/cnn_model.pth',
        'nn': 'training_files/simple_nn_model.pth',
        # Add other model paths as needed
    }
    
    # Load models
    demo.load_models(model_paths)
    
    if not demo.models:
        print("❌ No models loaded! Please train models first.")
        return
    
    # Demo menu
    while True:
        print("\n🎮 Demo Options:")
        print("1. 📹 Live webcam demo")
        print("2. 📸 Process image sequence")
        print("3. 📊 View session analytics")
        print("4. 🔄 Reset session")
        print("5. 🚪 Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            duration = int(input("Enter duration in seconds (default 60): ") or "60")
            demo.process_video_stream(duration=duration)
        
        elif choice == '2':
            # Example with test images
            test_dir = "archive/asl_alphabet_test"
            if os.path.exists(test_dir):
                # Get some sample images
                sample_images = []
                for class_dir in os.listdir(test_dir)[:5]:  # First 5 classes
                    class_path = os.path.join(test_dir, class_dir)
                    if os.path.isdir(class_path):
                        images = [f for f in os.listdir(class_path) if f.endswith('.jpg')][:2]
                        for img in images:
                            sample_images.append(os.path.join(class_path, img))
                
                if sample_images:
                    demo.process_image_sequence(sample_images)
                else:
                    print("⚠️ No test images found!")
            else:
                print("⚠️ Test directory not found!")
        
        elif choice == '3':
            demo.plot_session_analytics()
        
        elif choice == '4':
            demo.reset_sentence()
            print("🔄 Session reset!")
        
        elif choice == '5':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice!")


if __name__ == "__main__":
    main()
