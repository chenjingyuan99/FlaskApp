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

    # Clean and split text into words containing only English letters
    raw_words = T.split()
    stop_words = set(p.lower() for p in P)
    filtered_words = []
    removed_count = 0

    for word in raw_words:
        # Keep only English letters in each word
        word_clean = re.sub(r'[^a-zA-Z]', '', word)
        if word_clean.lower() in stop_words or not word_clean:
            if word_clean:
                removed_count += 1
        else:
            filtered_words.append(word_clean)
    filtered_text = ' '.join(filtered_words)

    def is_ascii_alpha(s):
        return all('a' <= c <= 'z' or 'A' <= c <= 'Z' for c in s)

    bigrams = []
    for i, word in enumerate(filtered_words):
        if word and any(word.lower().startswith(char.lower()) for char in S):
            if i > 0:
                bigram_prev = f"{filtered_words[i-1]} {word}"
                if is_ascii_alpha(filtered_words[i-1]) and is_ascii_alpha(word):
                    bigrams.append(bigram_prev)
            if i < len(filtered_words) - 1:
                bigram_next = f"{word} {filtered_words[i+1]}"
                if is_ascii_alpha(word) and is_ascii_alpha(filtered_words[i+1]):
                    bigrams.append(bigram_next)

    return jsonify({
        'removed_count': removed_count,
        'filtered_text': filtered_text,
        'bigrams': bigrams if bigrams else ['No bigrams found']
    })

if __name__ == '__main__':
    app.run(debug=True,port=5670)
