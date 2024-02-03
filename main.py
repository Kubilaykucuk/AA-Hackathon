from transformers import BertTokenizerFast, EncoderDecoderModel
import tkinter as tk
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import random
import numpy as np

# Download the Punkt tokenizer models (do this once)
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('turkish'))

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    
    sent1 = [w.lower() for w in word_tokenize(sent1) if w not in stopwords]
    sent2 = [w.lower() for w in word_tokenize(sent2) if w not in stopwords]
    
    all_words = list(set(sent1 + sent2))
    
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
    
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    
    return 1 - cosine_similarity([vector1], [vector2])[0][0]

def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    
    return similarity_matrix

def textrank(text, stopwords=stop_words):
    sentences = sent_tokenize(text)
    
    similarity_matrix = build_similarity_matrix(sentences, stopwords)
    
    similarity_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(similarity_graph)
    
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    return ranked_sentences

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

# Initialize the tokenizer and model for Turkish summarization
tokenizer = BertTokenizerFast.from_pretrained("mrm8488/bert2bert_shared-turkish-summarization")
model = EncoderDecoderModel.from_pretrained("mrm8488/bert2bert_shared-turkish-summarization")

def rearrange_sentences(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text, language='turkish')
    
    # Randomly shuffle the sentences
    random.shuffle(sentences)
    
    # Join the shuffled sentences back into a single string
    rearranged_text = ' '.join(sentences)
    return rearranged_text

def generate_headlines(text, num_headlines=3):
    # Prepare the text for the model
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    headlines = []
    settings = [
        {"temperature": 0.001, "top_k": 10, "top_p": 0.9},  # More certain, less random
        {"temperature": 0.9, "top_k": 20, "top_p": 0.92}, # A bit more random
        {"temperature": 1.1, "top_k": 30, "top_p": 0.94}, # Even more random
    ]

    # Extract quoted sentences from the text
    quoted_sentences = extract_quoted_sentences(text)

    # Perform TextRank on the given text
    ranked_sentences = textrank(text)

    for setting in settings:
        # Generate the summary with varying temperature to introduce diversity
        summary_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=80, min_length=5, do_sample=True, **setting, num_beams=4, num_return_sequences=1, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Post-process the summary to create a headline
        headline = summary.capitalize()
        if headline[-1] in ['.', ',', ';', ':']:
            headline = headline[:-1]

        first_period_pos = headline.find('.')
        if first_period_pos != -1:
            # Truncate the headline after the first period, including the period itself
            headline = headline[:first_period_pos + 1]
        
        headlines.append(headline)

    return headlines, quoted_sentences, ranked_sentences

def display_headlines(headlines, quoted_sentences, ranked_sentences):
    # Clear previous output
    for widget in output_frame.winfo_children():
        widget.destroy()

    # Display each headline in a separate label
    for i, headline in enumerate(headlines, 1):
        text_widget = tk.Text(output_frame, height=4, width=100)
        text_widget.pack()

        # Insert "Başlık" part with blue color and size 20
        text_widget.tag_configure("baslik_tag", foreground="blue", font=("Helvetica", 15))
        text_widget.insert("end", f"Özet {i}: ", "baslik_tag")

        # Insert headline part with black color, bold style, and size 15
        text_widget.tag_configure("headline_tag", foreground="black", font=("Helvetica", 12, "bold"))
        text_widget.insert("end", headline, "headline_tag")

        # Make the text widget read-only
        text_widget.config(state="disabled")
    for idx, (score, sentence) in enumerate(ranked_sentences[:3]):
        text_widget = tk.Text(output_frame, height=4, width=100)
        text_widget.pack()
        if sentence in quoted_sentences:
            text_widget.tag_configure("baslik_tag_spot", foreground="blue", font=("Helvetica", 15))
            text_widget.insert("end", f"Spot {idx+1}: ", "baslik_tag_spot")
            text_widget.tag_configure("spot_tag", foreground="black", font=("Helvetica", 12, "bold"))
            text_widget.insert("end", f"\"{sentence}\"", "spot_tag")
        else:
            text_widget.tag_configure("baslik_tag_spot", foreground="blue", font=("Helvetica", 15))
            text_widget.insert("end", f"Spot {idx+1}: ", "baslik_tag_spot")
            text_widget.tag_configure("spot_tag", foreground="black", font=("Helvetica", 12, "bold"))
            text_widget.insert("end", f"{sentence}", "spot_tag")
        

def generate_and_display_headlines(modify_text=False):
    input_text = text_box.get("1.0", "end-1c")  # Get text from textbox
    
    if modify_text:
        input_text = rearrange_sentences(input_text)  # Rearrange sentences if modify_text is True
    
    headlines, quoted_sentences, ranked_sentences = generate_headlines(input_text)  # Generate headlines
    
    display_headlines(headlines, quoted_sentences, ranked_sentences)  # Display headlines

# Initialize the main Tkinter window
window = tk.Tk()
window.title("Özet ve Spot Oluşturucu")
window.geometry("800x700")

# Create a textbox for input
text_box = tk.Text(window, height=10, width=50)
text_box.pack()

# Create a button to trigger headline generation and display
generate_button = tk.Button(window, text="Özet ve Spot Oluştur", command=lambda: generate_and_display_headlines(modify_text=False))
generate_button.pack()

# Create a button to rearrange sentences and generate headlines
rearrange_button = tk.Button(window, text="Yeni Özet Oluştur", command=lambda: generate_and_display_headlines(modify_text=True))
rearrange_button.pack()

# Frame to hold output headlines
output_frame = tk.Frame(window)
output_frame.pack()

# Start the GUI loop
window.mainloop()