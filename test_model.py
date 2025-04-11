from streamlit_app import load_model, classify_text

def test_model():
    # Test text
    test_text = "This is a test sentence to verify if the model works correctly."
    
    try:
        print("Loading model...")
        model_path = "llama_model"
        tokenizer, model, device = load_model(model_path)
        print("Model loaded successfully!")
        
        print("\nPerforming classification on test text:")
        print(f"Text: {test_text}")
        
        predicted_class, confidence = classify_text(test_text, tokenizer, model, device)
        
        print("\nClassification Results:")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence*100:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        return False

if __name__ == "__main__":
    success = test_model()
    if success:
        print("\nTest completed successfully! The model is working correctly.")
    else:
        print("\nTest failed! Please check the error messages above.")