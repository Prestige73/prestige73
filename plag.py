from flask import Flask, request, render_template
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def load_matched_contents():
    with open('matched_contents.txt', 'r') as file:
        return [line.strip() for line in file.readlines()]

def calculate_similarity(query, matched_contents):
    processed_query = preprocess_text(query)
    processed_contents = [preprocess_text(content) for content in matched_contents]

    vectorizer = CountVectorizer().fit_transform([processed_query] + processed_contents)
    vectors = vectorizer.toarray()

    cosine_matrix = cosine_similarity(vectors)
    return cosine_matrix[0][1:]

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['text']
        matched_contents = load_matched_contents()
        similarities = calculate_similarity(text, matched_contents)

        plagiarism_percentage = max(similarities) * 100
        matched_sources = [matched_contents[i] for i in range(len(similarities)) if similarities[i] > 0]

        return render_template('result.html', 
                               plagiarism_percentage=plagiarism_percentage,
                               matched_sources=matched_sources)
    return render_template('index.html')

if __name__ == '__main__':
    app.run()