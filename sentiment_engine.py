import re
import nltk

import ssl

# Ensure SSL context bypass for NLTK dataset downloading on Windows networks
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
except Exception:
    nltk.download('vader_lexicon', quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()

class HybridSentimentEngine:
    def __init__(self):
        self.vader = vader
        self._trained = False
        self.classifier = None
        self.word_features = []

    def preprocess(self, text):
        """Clean and tokenize text."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|\#', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    def analyze_vader(self, text):
        """Perform VADER valence analysis."""
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
        Train a Naive Bayes classifier on provided (text, label) tuples.
        """
        from collections import Counter
        from nltk.classify import NaiveBayesClassifier

        cleaned_docs = []
        all_words = []

        for text, label in training_data:
            cleaned = self.preprocess(text)
            words = cleaned.split()
            cleaned_docs.append((words, label))
            all_words.extend(words)

        # Select top N most common word features
        word_counts = Counter(all_words)
        self.word_features = [word for word, _ in word_counts.most_common(top_n_features)]

        def extract_features(words):
            words_set = set(words)
            return {f"contains({word})": (word in words_set) for word in self.word_features}

        training_set = [(extract_features(words), label) for words, label in cleaned_docs]
        self.classifier = NaiveBayesClassifier.train(training_set)
        self._trained = True

    def classify_ml(self, text):
        """Classify text using the trained Naive Bayes classifier."""
        if not self._trained or not self.classifier:
            return {'label': 'Uncertain', 'confidence': 0.5}

        cleaned = self.preprocess(text).split()
        words_set = set(cleaned)
        features = {f"contains({word})": (word in words_set) for word in self.word_features}
        
        prob_dist = self.classifier.prob_classify(features)
        pred_label = prob_dist.max()
        confidence = round(prob_dist.prob(pred_label), 3)

        return {
            'label': pred_label,
            'confidence': confidence
        }

    def analyze(self, text):
        """Unified hybrid sentiment evaluation."""
        vader_res = self.analyze_vader(text)
        ml_res = self.classify_ml(text)

        # Consensus determination
        if self._trained:
            if vader_res['label'] == ml_res['label']:
                consensus = vader_res['label']
            elif vader_res['compound'] > 0.4:
                consensus = 'Positive'
            elif vader_res['compound'] < -0.4:
                consensus = 'Negative'
            else:
                consensus = ml_res['label']
        else:
            consensus = vader_res['label']

        return {
            'text': text,
            'consensus_label': consensus,
            'vader': vader_res,
            'ml': ml_res
        }
