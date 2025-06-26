from flask import Flask, render_template, request, redirect, url_for
import os
import re
import math
from collections import defaultdict, Counter
import time
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
import difflib
from itertools import permutations
import string
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

app = Flask(__name__)

class FuzzyBM25SearchEngine:

    def __init__(self,  k1=1.5, b=0.75):

        self.connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        self.container_name = os.getenv('BLOB_CONTAINER_NAME')
        self.stopwords_file = 'StopWords.txt'
        self.documents = {}
        self.stopwords = set()
        self.blob_service_client = None
        
        # BM25 parameters
        self.k1 = k1
        self.b = b
        
        # BM25 specific data structures
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        self.corpus_size = 0
        self.doc_names = []
        
        # Fuzzy matching parameters
        self.vocabulary = set()  # All unique words in corpus
        self.fuzzy_threshold = 0.6  # Minimum similarity for fuzzy matches
        self.max_edit_distance = 2  # Maximum edit distance for corrections
        
        if self.connection_string:
            try:
                self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
                self.load_stopwords()
                self.load_documents()
                self.build_bm25_index()
                self.build_vocabulary()
                print(f"Successfully loaded {len(self.documents)} documents with fuzzy BM25 indexing")
            except Exception as e:
                print(f"Error connecting to Azure Blob Storage: {e}")
        else:
            print("Azure Storage connection string not found.")
    
    def read_blob_content(self, blob_name):
        """Read content from Azure Blob Storage"""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=blob_name
            )
            blob_data = blob_client.download_blob()
            content = blob_data.readall().decode('utf-8', errors='ignore')
            return content
        except Exception as e:
            print(f"Error reading blob {blob_name}: {e}")
            return None
    
    def list_blobs(self):
        """List all blobs in the container"""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            blob_list = container_client.list_blobs()
            return [blob.name for blob in blob_list if blob.name.endswith('.txt')]
        except Exception as e:
            print(f"Error listing blobs: {e}")
            return []
    
    def load_stopwords(self):
        """Load stop words from Azure Blob Storage"""
        try:
            content = self.read_blob_content(self.stopwords_file)
            if content:
                words = re.findall(r'"([^"]*)"', content.lower())
                for word in words:
                    clean_word = word.strip().lower()
                    if clean_word:
                        self.stopwords.add(clean_word)
                print(f"Loaded {len(self.stopwords)} stop words")
        except Exception as e:
            print(f"Error loading stopwords: {e}")
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        text = text.lower()
        text = ''.join(char for char in text if ord(char) < 128)
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        words = [word for word in words if word and word not in self.stopwords and len(word) > 1]
        return words
    
    def levenshtein_distance(self, s1, s2):
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def get_fuzzy_matches(self, word):
        """Find fuzzy matches for a word in the vocabulary"""
        matches = []
        
        # Exact match first
        if word in self.vocabulary:
            matches.append((word, 1.0, 'exact'))
        
        # Find close matches using difflib
        close_matches = difflib.get_close_matches(word, self.vocabulary, n=5, cutoff=self.fuzzy_threshold)
        for match in close_matches:
            if match != word:  # Skip exact matches already added
                similarity = difflib.SequenceMatcher(None, word, match).ratio()
                matches.append((match, similarity, 'fuzzy'))
        
        # Find matches with small edit distance
        for vocab_word in self.vocabulary:
            if abs(len(word) - len(vocab_word)) <= self.max_edit_distance:
                edit_dist = self.levenshtein_distance(word, vocab_word)
                if 0 < edit_dist <= self.max_edit_distance:
                    similarity = 1.0 - (edit_dist / max(len(word), len(vocab_word)))
                    if similarity >= self.fuzzy_threshold:
                        # Check if not already added
                        if not any(match[0] == vocab_word for match in matches):
                            matches.append((vocab_word, similarity, 'edit_distance'))
        
        # Handle letter transposition (swap adjacent letters)
        for i in range(len(word) - 1):
            transposed = word[:i] + word[i+1] + word[i] + word[i+2:]
            if transposed in self.vocabulary and transposed != word:
                if not any(match[0] == transposed for match in matches):
                    matches.append((transposed, 0.9, 'transposition'))
        
        # Sort by similarity score (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:5]  # Return top 5 matches
    
    def expand_query_with_fuzzy_matches(self, query_words):
        """Expand query words with fuzzy matches"""
        expanded_query = []
        fuzzy_info = {}
        
        for word in query_words:
            fuzzy_matches = self.get_fuzzy_matches(word)
            
            if fuzzy_matches:
                # Add original word info
                fuzzy_info[word] = {
                    'original': word,
                    'matches': fuzzy_matches,
                    'expanded_terms': [match[0] for match in fuzzy_matches]
                }
                
                # Add all fuzzy matches to expanded query
                for match, similarity, match_type in fuzzy_matches:
                    expanded_query.append({
                        'term': match,
                        'original': word,
                        'similarity': similarity,
                        'type': match_type
                    })
            else:
                # No matches found, keep original
                expanded_query.append({
                    'term': word,
                    'original': word,
                    'similarity': 1.0,
                    'type': 'no_match'
                })
                fuzzy_info[word] = {
                    'original': word,
                    'matches': [],
                    'expanded_terms': [word]
                }
        
        return expanded_query, fuzzy_info
    
    def load_documents(self):
        """Load all text documents from Azure Blob Storage"""
        blob_names = self.list_blobs()
        
        for blob_name in blob_names:
            if blob_name != self.stopwords_file:
                try:
                    content = self.read_blob_content(blob_name)
                    if content:
                        lines = content.split('\n')
                        cleaned_words = self.clean_text(content)
                        self.documents[blob_name] = {
                            'content': content,
                            'lines': lines,
                            'words': cleaned_words,
                            'total_words': len(cleaned_words)
                        }
                        print(f"Loaded {blob_name}: {len(cleaned_words)} words")
                except Exception as e:
                    print(f"Error loading {blob_name}: {e}")
    
    def build_vocabulary(self):
        """Build vocabulary from all documents"""
        for doc_data in self.documents.values():
            self.vocabulary.update(doc_data['words'])
        print(f"Built vocabulary with {len(self.vocabulary)} unique terms")
    
    def build_bm25_index(self):
        """Build BM25 index for all documents"""
        self.corpus_size = len(self.documents)
        self.doc_names = list(self.documents.keys())
        
        nd = {}
        total_doc_length = 0
        
        for doc_name in self.doc_names:
            doc_words = self.documents[doc_name]['words']
            doc_length = len(doc_words)
            self.doc_len.append(doc_length)
            total_doc_length += doc_length
            
            term_freqs = {}
            for word in doc_words:
                term_freqs[word] = term_freqs.get(word, 0) + 1
            
            self.doc_freqs.append(term_freqs)
            
            for word in term_freqs:
                nd[word] = nd.get(word, 0) + 1
        
        self.avgdl = total_doc_length / self.corpus_size if self.corpus_size > 0 else 0
        
        for word, doc_freq in nd.items():
            idf = math.log((self.corpus_size - doc_freq + 0.5) / (doc_freq + 0.5))
            self.idf[word] = max(idf, 0)
        
        print(f"Built BM25 index with {len(self.idf)} unique terms")
    
    def calculate_fuzzy_bm25_score(self, expanded_query, doc_index):
        """Calculate BM25 score with fuzzy matching"""
        score = 0.0
        doc_freqs = self.doc_freqs[doc_index]
        doc_length = self.doc_len[doc_index]
        
        # Group expanded terms by original query word
        original_terms = {}
        for item in expanded_query:
            original = item['original']
            if original not in original_terms:
                original_terms[original] = []
            original_terms[original].append(item)
        
        # Calculate score for each original query term
        for original_word, term_variants in original_terms.items():
            max_term_score = 0.0
            
            # Find the best matching variant for this original term
            for variant in term_variants:
                term = variant['term']
                similarity = variant['similarity']
                
                if term in self.idf:
                    tf = doc_freqs.get(term, 0)
                    if tf > 0:
                        idf = self.idf[term]
                        numerator = tf * (self.k1 + 1)
                        denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avgdl))
                        
                        term_score = idf * (numerator / denominator) * similarity
                        max_term_score = max(max_term_score, term_score)
            
            score += max_term_score
        
        return score
    
    def search(self, query):
        """Search with fuzzy matching and BM25 scoring"""
        start_time = time.time()
        
        query_words = self.clean_text(query)
        if not query_words:
            return [], 0, {}
        
        # Expand query with fuzzy matches
        expanded_query, fuzzy_info = self.expand_query_with_fuzzy_matches(query_words)
        
        results = []
        
        for i, doc_name in enumerate(self.doc_names):
            bm25_score = self.calculate_fuzzy_bm25_score(expanded_query, i)
            
            if bm25_score > 0:
                doc_data = self.documents[doc_name]
                
                # Calculate metrics
                query_words_found = set()
                term_occurrences = 0
                fuzzy_matches_used = []
                
                for item in expanded_query:
                    term = item['term']
                    original = item['original']
                    if term in self.doc_freqs[i]:
                        query_words_found.add(original)
                        term_occurrences += self.doc_freqs[i][term]
                        if item['type'] != 'exact':
                            fuzzy_matches_used.append({
                                'original': original,
                                'matched': term,
                                'type': item['type'],
                                'similarity': item['similarity']
                            })
                
                matching_rate = (len(query_words_found) / len(query_words)) * 100
                matching_lines = self.find_fuzzy_matching_lines(doc_name, expanded_query)
                
                results.append({
                    'document': doc_name,
                    'bm25_score': bm25_score,
                    'matching_rate': matching_rate,
                    'total_words': doc_data['total_words'],
                    'term_occurrences': term_occurrences,
                    'query_words_found': len(query_words_found),
                    'total_query_words': len(query_words),
                    'fuzzy_matches_used': fuzzy_matches_used,
                    'matching_lines': matching_lines[:10]
                })
        
        results.sort(key=lambda x: x['bm25_score'], reverse=True)
        search_time = time.time() - start_time
        
        return results, search_time, fuzzy_info
    
    def find_fuzzy_matching_lines(self, doc_name, expanded_query):
        """Find lines with fuzzy matches"""
        lines = self.documents[doc_name]['lines']
        matching_lines = []
        
        # Create lookup for all terms
        all_terms = set()
        term_to_original = {}
        for item in expanded_query:
            all_terms.add(item['term'])
            term_to_original[item['term']] = item['original']
        
        for line_num, line in enumerate(lines, 1):
            line_words = self.clean_text(line)
            
            matches_in_line = []
            for word in line_words:
                if word in all_terms:
                    original = term_to_original[word]
                    matches_in_line.append((word, original))
            
            if matches_in_line:
                highlighted_line = line
                for matched_word, original_word in matches_in_line:
                    pattern = re.compile(re.escape(matched_word), re.IGNORECASE)
                    if matched_word == original_word:
                        highlighted_line = pattern.sub(f'<mark>{matched_word}</mark>', highlighted_line)
                    else:
                        highlighted_line = pattern.sub(f'<mark class="fuzzy">{matched_word}</mark>', highlighted_line)
                
                matching_lines.append({
                    'line_number': line_num,
                    'content': highlighted_line,
                    'matches': [original for _, original in matches_in_line]
                })
        
        return matching_lines
    
    def analyze_letter_frequency(self, document_name):
        """Analyze letter frequency for a specific document"""
        if document_name not in self.documents:
            return None, f"Document '{document_name}' not found"
        
        # Get document content
        content = self.documents[document_name]['content'].lower()
        
        # Initialize letter counts for all 26 letters
        letter_counts = {letter: 0 for letter in string.ascii_lowercase}
        total_letters = 0
        
        # Count each letter
        for char in content:
            if char.isalpha() and char in letter_counts:
                letter_counts[char] += 1
                total_letters += 1
        
        # Calculate percentages
        letter_frequencies = {}
        for letter, count in letter_counts.items():
            percentage = (count / total_letters * 100) if total_letters > 0 else 0
            letter_frequencies[letter] = {
                'count': count,
                'percentage': round(percentage, 2)
            }
        
        # Sort by frequency (highest first)
        sorted_frequencies = dict(sorted(letter_frequencies.items(), 
                                       key=lambda x: x[1]['count'], 
                                       reverse=True))
        
        return {
            'document': document_name,
            'total_letters': total_letters,
            'total_words': self.documents[document_name]['total_words'],
            'frequencies': sorted_frequencies
        }, None

# Initialize fuzzy search engine
search_engine = FuzzyBM25SearchEngine()

@app.route('/')
def index():
    return render_template('index.html', 
                         total_documents=len(search_engine.documents),
                         document_names=list(search_engine.documents.keys()),
                         connection_status="Connected with Azure blob" if search_engine.blob_service_client else "Not connected")

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query', '').strip()
    if not query:
        return redirect(url_for('index'))
    
    results, search_time, fuzzy_info = search_engine.search(query)
    
    return render_template('results.html', 
                         query=query,
                         results=results,
                         search_time=search_time,
                         total_results=len(results),
                         fuzzy_info=fuzzy_info)

@app.route('/letter_frequency', methods=['POST'])
def letter_frequency():
    document_name = request.form.get('document_name', '').strip()
    if not document_name:
        return redirect(url_for('index'))
    
    analysis_result, error = search_engine.analyze_letter_frequency(document_name)
    
    if error:
        return render_template('frequency_results.html', 
                             error=error,
                             document_name=document_name,
                             available_documents=list(search_engine.documents.keys()))
    
    return render_template('frequency_results.html', 
                         analysis=analysis_result)   

if __name__ == '__main__':
    app.run(debug=True)
