import os
import argparse
from src.webcam_capture import WebcamCapture
from src.translate import create_translator, LANGUAGE_CODES
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='ASL Alphabet Interpreter')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth',
                      help='Path to the trained model checkpoint')
    parser.add_argument('--confidence', type=float, default=0.5,
                      help='Confidence threshold for predictions')
    parser.add_argument('--target_lang', type=str, default='es',
                      help='Target language code for translation')
    parser.add_argument('--train', action='store_true',
                      help='Train the model before running inference')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Train the model if requested
    if args.train:
        print("Training the model...")
        from src.train import main as train_main
        train_main()
        print("Training completed!")
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found at {args.model_path}")
        print("Please train the model first using --train")
        return
    
    # Initialize translator
    translator = create_translator(target_lang=args.target_lang)
    
    # Initialize webcam capture
    webcam = WebcamCapture(
        model_path=args.model_path,
        confidence_threshold=args.confidence
    )
    
    print("\nASL Alphabet Interpreter")
    print("----------------------")
    print(f"Target language: {args.target_lang}")
    print("Press 'q' to quit")
    print("Hold your hand in the green box to make signs")
    print("Hold a sign for 1 second to add it to the word")
    print("The word will be translated automatically")
    print("\nStarting...")
    
    try:
        while True:
            # Get current word
            current_word = webcam.get_current_word()
            
            # If we have a word, translate it
            if current_word:
                try:
                    translation = translator.translate(current_word)
                    print(f"\nWord: {current_word}")
                    print(f"Translation: {translation}")
                    webcam.clear_word()  # Clear the word after translation
                except Exception as e:
                    print(f"Translation error: {e}")
            
            # Run webcam capture
            webcam.run()
            
            # Break if 'q' was pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 