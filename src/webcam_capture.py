import cv2
import numpy as np
import time
from PIL import Image
from src.inference import ASLPredictor

class WebcamCapture:
    def __init__(self, model_path='checkpoints/best_model.pth', confidence_threshold=0.5):
        """
        Initialize the webcam capture
        
        Args:
            model_path (str): Path to the trained model checkpoint
            confidence_threshold (float): Confidence threshold for predictions
        """
        self.predictor = ASLPredictor(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize variables for letter accumulation
        self.current_letter = None
        self.letter_start_time = None
        self.letter_duration = 1.0  # seconds to hold a sign
        self.current_word = []
        self.last_prediction_time = 0
        self.prediction_cooldown = 0.5  # seconds between predictions
    
    def preprocess_frame(self, frame):
        """
        Preprocess a frame for ASL recognition
        
        Args:
            frame: OpenCV frame
            
        Returns:
            numpy.ndarray: Preprocessed frame
        """
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        
        return frame_resized
    
    def draw_hand_region(self, frame):
        """
        Draw a rectangle to indicate the hand region
        
        Args:
            frame: OpenCV frame
            
        Returns:
            numpy.ndarray: Frame with hand region indicator
        """
        height, width = frame.shape[:2]
        x1 = int(width * 0.25)
        y1 = int(height * 0.25)
        x2 = int(width * 0.75)
        y2 = int(height * 0.75)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame
    
    def update_letter_buffer(self, predicted_letter, confidence):
        """
        Update the letter buffer based on prediction
        
        Args:
            predicted_letter (str): Predicted ASL letter
            confidence (float): Prediction confidence
        """
        current_time = time.time()
        
        if predicted_letter is None:
            if self.current_letter is not None:
                if current_time - self.letter_start_time > self.letter_duration:
                    self.current_word.append(self.current_letter)
                    self.current_letter = None
                    self.letter_start_time = None
            return
        
        if predicted_letter != self.current_letter:
            if self.current_letter is None:
                self.current_letter = predicted_letter
                self.letter_start_time = current_time
            elif current_time - self.letter_start_time > self.letter_duration:
                self.current_word.append(self.current_letter)
                self.current_letter = predicted_letter
                self.letter_start_time = current_time
    
    def run(self):
        """
        Run the webcam capture loop
        """
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Draw hand region
                frame = self.draw_hand_region(frame)
                
                # Get current time
                current_time = time.time()
                
                # Process frame if enough time has passed since last prediction
                if current_time - self.last_prediction_time >= self.prediction_cooldown:
                    # Preprocess frame
                    processed_frame = self.preprocess_frame(frame)
                    
                    # Get prediction
                    predicted_letter, confidence = self.predictor.predict(processed_frame, self.confidence_threshold)
                    
                    # Update letter buffer
                    self.update_letter_buffer(predicted_letter, confidence)
                    
                    # Update last prediction time
                    self.last_prediction_time = current_time
                
                # Display current word
                current_word = ''.join(self.current_word)
                cv2.putText(frame, f"Word: {current_word}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display current letter and confidence
                if self.current_letter:
                    cv2.putText(frame, f"Letter: {self.current_letter}", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('ASL Recognition', frame)
                
                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
    
    def get_current_word(self):
        """
        Get the current word being spelled
        
        Returns:
            str: Current word
        """
        return ''.join(self.current_word)
    
    def clear_word(self):
        """Clear the current word buffer"""
        self.current_word = []
        self.current_letter = None
        self.letter_start_time = None 