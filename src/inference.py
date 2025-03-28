import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from src.model import load_model

class ASLPredictor:
    def __init__(self, model_path, device=None):
        """
        Initialize the ASL predictor
        
        Args:
            model_path (str): Path to the trained model checkpoint
            device (str, optional): Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load class names
        self.classes = sorted([chr(i) for i in range(65, 91)])  # A-Z
    
    def preprocess_image(self, image):
        """
        Preprocess a single image for inference
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Apply transforms
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0).to(self.device)
    
    def predict(self, image, threshold=0.5):
        """
        Predict the ASL sign from an image
        
        Args:
            image: PIL Image or numpy array
            threshold (float): Confidence threshold for prediction
            
        Returns:
            tuple: (predicted_letter, confidence)
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            confidence = confidence.item()
            predicted_letter = self.classes[predicted.item()]
            
            if confidence < threshold:
                return None, confidence
            
            return predicted_letter, confidence
    
    def predict_batch(self, images, threshold=0.5):
        """
        Predict ASL signs from a batch of images
        
        Args:
            images (list): List of PIL Images or numpy arrays
            threshold (float): Confidence threshold for prediction
            
        Returns:
            list: List of (predicted_letter, confidence) tuples
        """
        results = []
        for image in images:
            pred, conf = self.predict(image, threshold)
            results.append((pred, conf))
        return results

def create_predictor(model_path='checkpoints/best_model.pth'):
    """
    Factory function to create an ASL predictor
    
    Args:
        model_path (str): Path to the trained model checkpoint
        
    Returns:
        ASLPredictor: The initialized predictor
    """
    return ASLPredictor(model_path) 