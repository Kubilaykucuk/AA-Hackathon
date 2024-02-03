import nltk
from nltk.tokenize import sent_tokenize

# Ensure you have the necessary NLTK data
nltk.download('punkt')

def extract_quoted_sentences(text):
    quoted_sentences = []
    quote_chars = ['"', '“', '”']  # Handling different types of quote characters
    start = None  # Start index of the quote

    # Iterate through each character in the text
    for i, char in enumerate(text):
        # If the current character is a quote and start is None, it's the start of a quote
        if char in quote_chars and start is None:
            start = i + 1  # Set start to the character after the quote
        # If the current character is a quote and start is not None, it's the end of a quote
        elif char in quote_chars and start is not None:
            quoted_text = text[start:i]  # Extract the quoted text
            # Tokenize the quoted text into sentences
            sentences = sent_tokenize(quoted_text, language='turkish')  # Specify language if necessary
            quoted_sentences.extend(sentences)  # Add the individual sentences to the list
            start = None  # Reset start to None for the next quote

    return quoted_sentences

# Example text in Turkish with quotes
text = """Merhaba, bu bir örnek metindir. "Bu bir alıntıdır. Ve önemlidir. Bu da bir alıntıdır" TextRank algoritması, belirli bir metindeki cümleleri sıralamak için kullanılabilir. "Bu da başka bir alıntıdır. Ve bu da önemli." Bu örnek, temel bir TextRank uygulamasını göstermektedir."""

# Extract quoted sentences from the text
quoted_sentences = extract_quoted_sentences(text)

# Print the extracted quoted sentences
for sentence in quoted_sentences:
    print(f"Quoted Sentence: {sentence}")
