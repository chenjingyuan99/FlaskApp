from flask import Flask, render_template, request, jsonify
import re
from collections import Counter

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_task10', methods=['POST'])
def process_task10():
    data = request.json
    S = data.get('S', '')
    T = data.get('T', '')
    C = data.get('C', '')
    
    # Count characters in S that appear in T
    char_counts = {}
    char_frequencies = {}
    total_chars = len(T)
    
    for char in S:
        count = T.count(char)
        char_counts[char] = count
        char_frequencies[char] = count / total_chars if total_chars > 0 else 0
    
    # Replace characters
    modified_T = T
    for char in S:
        modified_T = modified_T.replace(char, C)
    
    return jsonify({
        'counts': char_counts,
        'frequencies': char_frequencies,
        'modified_text': modified_T
    })

@app.route('/process_task11', methods=['POST'])
def process_task11():
    data = request.json
    S = data.get('S', '')
    T = data.get('T', '')
    
    words = T.split()
    word_count = len(words)
    
    words_by_char = {}
    for char in S:
        words_starting_with_char = [word for word in words if word.lower().startswith(char.lower())]
        words_by_char[char] = words_starting_with_char
    
    return jsonify({
        'word_count': word_count,
        'words_by_char': words_by_char
    })

@app.route('/process_task12', methods=['POST'])
def process_task12():
    data = request.json
    S = data.get('S', '')
    T = data.get('T', '')
    P = data.get('P', [])
    
    words = T.split()
    original_count = len(words)
    
    # Remove stop words
    filtered_words = [word for word in words if word.lower() not in [p.lower() for p in P]]
    removed_count = original_count - len(filtered_words)
    filtered_text = ' '.join(filtered_words)
    
    # Find bigrams for words starting with letters in S
    bigrams = []
    for i, word in enumerate(filtered_words):
        if any(word.lower().startswith(char.lower()) for char in S):
            # Preceding bigram
            if i > 0:
                bigrams.append(f"{filtered_words[i-1]} {word}")
            # Following bigram
            if i < len(filtered_words) - 1:
                bigrams.append(f"{word} {filtered_words[i+1]}")
    
    return jsonify({
        'removed_count': removed_count,
        'filtered_text': filtered_text,
        'bigrams': bigrams
    })

if __name__ == '__main__':
    app.run(debug=True,port=5670)
