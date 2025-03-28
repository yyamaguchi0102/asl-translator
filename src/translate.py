from transformers import MarianMTModel, MarianTokenizer

class Translator:
    def __init__(self, source_lang='en', target_lang='es'):
        """
        Initialize the translator
        
        Args:
            source_lang (str): Source language code (default: 'en' for English)
            target_lang (str): Target language code (default: 'es' for Spanish)
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # Load model and tokenizer
        model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
    
    def translate(self, text):
        """
        Translate text from source language to target language
        
        Args:
            text (str): Text to translate
            
        Returns:
            str: Translated text
        """
        # Tokenize the text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        
        # Generate translation
        outputs = self.model.generate(**inputs)
        
        # Decode the translation
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return translated_text

def create_translator(source_lang='en', target_lang='es'):
    """
    Factory function to create a translator
    
    Args:
        source_lang (str): Source language code
        target_lang (str): Target language code
        
    Returns:
        Translator: The initialized translator
    """
    return Translator(source_lang, target_lang)

# Common language codes
LANGUAGE_CODES = {
    'english': 'en',
    'spanish': 'es',
    'french': 'fr',
    'german': 'de',
    'italian': 'it',
    'portuguese': 'pt',
    'russian': 'ru',
    'chinese': 'zh',
    'japanese': 'ja',
    'korean': 'ko'
} 