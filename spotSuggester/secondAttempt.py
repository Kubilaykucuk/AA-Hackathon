import spacy
import networkx as nx
import numpy as np
from scipy.spatial.distance import cosine
from nltk.tokenize import sent_tokenize

# Load the Turkish model for spaCy
nlp = spacy.load("tr_core_news_sm")

def sentence_vector(sentence):
    """Generate sentence vector by averaging word vectors, ignoring out-of-vocabulary words and stopwords."""
    doc = nlp(sentence)
    vectors = [token.vector for token in doc if not token.is_stop and not token.is_punct and token.has_vector]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros((nlp.vocab.vectors_length,))

def sentence_similarity(vec1, vec2):
    """Calculate cosine similarity between two sentence vectors. Handle the case of zero vectors to avoid division by zero."""
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0
    return 1 - cosine(vec1, vec2)

def build_similarity_matrix(sentences):
    """Build similarity matrix based on cosine similarity between sentence vectors."""
    vectors = [sentence_vector(sentence) for sentence in sentences]
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(vectors[i], vectors[j])
    
    return similarity_matrix

def textrank(text):
    sentences = sent_tokenize(text)
    similarity_matrix = build_similarity_matrix(sentences)
    
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    return ranked_sentences

# Example text in Turkish
text = "Merhaba, bu bir örnek metindir. TextRank algoritması, belirli bir metindeki cümleleri sıralamak için kullanılabilir. Bu örnek, temel bir TextRank uygulamasını göstermektedir."

ranked_sentences = textrank(text)

for idx, (score, sentence) in enumerate(ranked_sentences):
    print(f"Rank: {idx+1}, Score: {score:.4f}, Sentence: {sentence}")
