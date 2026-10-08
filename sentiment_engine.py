import os
import re
import time
import tempfile
import ssl
import nltk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
temp_nltk_dir = os.path.join(tempfile.gettempdir(), 'sentipulse_nltk_data')

# Include local project nltk_data directory and OS temp directory
nltk.data.path.append(os.path.join(BASE_DIR, 'nltk_data'))
nltk.data.path.append(temp_nltk_dir)

# Bypass SSL verification if NLTK auto-downloads are triggered in restricted network environments
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Initialize VADER Sentiment Intensity Analyzer with fallback handling
vader = None
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
except Exception:
    try:
        os.makedirs(temp_nltk_dir, exist_ok=True)
        nltk.download('vader_lexicon', download_dir=temp_nltk_dir, quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        vader = SentimentIntensityAnalyzer()
    except Exception as err:
        print(f"VADER initialization notice: {err}")
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            vader = SentimentIntensityAnalyzer()
        except Exception:
            vader = None

# Comprehensive set of English stopwords for clean ML feature extraction
ENGLISH_STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'cannot', 'could',
    'couldn', 'did', 'didn', 'do', 'does', 'doesn', 'doing', 'don', 'down', 'during', 'each', 'few', 'for',
    'from', 'further', 'had', 'hadn', 'has', 'hasn', 'have', 'haven', 'having', 'he', 'her', 'here', 'hers',
    'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', 'it', 'its', 'itself',
    'just', 'me', 'more', 'most', 'mustn', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once',
    'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'shan', 'she', 'should',
    'shouldn', 'so', 'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then',
    'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was',
    'wasn', 'we', 'were', 'weren', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'with',
    'won', 'would', 'wouldn', 'you', 'your', 'yours', 'yourself', 'yourselves'
}

class HybridSentimentEngine:
    def __init__(self):
        self.vader = vader
        self._trained = False
        self.classifier = None
        self.word_features = []

    def preprocess(self, text):
        """
        Clean and tokenize text for Machine Learning feature extraction.
        Removes URLs, mentions, punctuation, numbers, and common stop words.
        """
        if not isinstance(text, str):
            return []
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+|\#', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = [w for w in text.split() if w not in ENGLISH_STOPWORDS and len(w) > 2]
        return tokens

    def analyze_vader(self, text):
        """Perform VADER valence analysis on raw input text."""
        if not self.vader:
            return {'pos': 0.0, 'neg': 0.0, 'neu': 1.0, 'compound': 0.0, 'label': 'Neutral'}
            
        scores = self.vader.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            label = 'Positive'
        elif compound <= -0.05:
            label = 'Negative'
        else:
            label = 'Neutral'

        return {
            'pos': round(scores['pos'], 3),
            'neg': round(scores['neg'], 3),
            'neu': round(scores['neu'], 3),
            'compound': round(compound, 4),
            'label': label
        }

    def train_ml_model(self, training_data, top_n_features=500):
        """
        Train Naive Bayes classifier on provided (text, label) pairs.
        Extracts informative unigrams after removing noise and stopwords.
        """
        from collections import Counter
        from nltk.classify import NaiveBayesClassifier

        cleaned_docs = []
        all_words = []

        for text, label in training_data:
            tokens = self.preprocess(text)
            cleaned_docs.append((tokens, label))
            all_words.extend(tokens)

        # Select top N most informative word features
        word_counts = Counter(all_words)
        self.word_features = [word for word, _ in word_counts.most_common(top_n_features)]

        def extract_features(tokens):
            tokens_set = set(tokens)
            return {f"contains({word})": (word in tokens_set) for word in self.word_features}

        training_set = [(extract_features(tokens), label) for tokens, label in cleaned_docs]
        self.classifier = NaiveBayesClassifier.train(training_set)
        self._trained = True

    def classify_ml(self, text):
        """Classify text using the trained Naive Bayes model."""
        if not self._trained or not self.classifier:
            return {'label': 'Uncertain', 'confidence': 0.5, 'matched_features': []}

        tokens = self.preprocess(text)
        tokens_set = set(tokens)
        
        matched = [w for w in self.word_features if w in tokens_set]
        features = {f"contains({word})": (word in tokens_set) for word in self.word_features}
        
        prob_dist = self.classifier.prob_classify(features)
        pred_label = prob_dist.max()
        confidence = round(prob_dist.prob(pred_label), 3)

        return {
            'label': pred_label,
            'confidence': confidence,
            'matched_features': matched
        }

    def analyze(self, text):
        """
        Unified hybrid sentiment analysis combining VADER valence scoring
        and Naive Bayes ML classification with execution metrics.
        """
        start_time = time.time()
        
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        # Limit text length to prevent CPU exhaustion
        text_processed = text[:5000]

        vader_res = self.analyze_vader(text_processed)
        ml_res = self.classify_ml(text_processed)

        # Hybrid Consensus Logic
        if self._trained and ml_res['label'] != 'Uncertain':
            if vader_res['label'] == ml_res['label']:
                consensus = vader_res['label']
            elif abs(vader_res['compound']) >= 0.4:
                # Strong VADER valence takes priority for highly expressive text
                consensus = vader_res['label']
            else:
                # Naive Bayes provides learned domain context for subtle expressions
                consensus = ml_res['label']
        else:
            consensus = vader_res['label']

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            'text': text,
            'consensus_label': consensus,
            'vader': vader_res,
            'ml': ml_res,
            'processing_time_ms': elapsed_ms
        }
