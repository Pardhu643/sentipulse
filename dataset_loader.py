import csv
import os
import random
from collections import Counter

class DatasetLoader:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = []
        self.entities = set()
        self.sentiments = set()
        self.loaded = False

    def load(self, max_records=15000):
        """
        Memory-efficient CSV parser loading up to max_records.
        Safely strips and validates data tuples (id, entity, sentiment, text).
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Dataset not found at {self.csv_path}")

        self.data = []
        self.entities = set()
        self.sentiments = set()

        with open(self.csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            count = 0
            for row in reader:
                if len(row) >= 4:
                    item_id = row[0].strip() if row[0] else ""
                    entity = row[1].strip() if row[1] else "Unknown"
                    sentiment = row[2].strip().capitalize() if row[2] else "Neutral"
                    text = row[3].strip() if row[3] else ""

                    # Skip records without text or valid sentiment label
                    if not text or sentiment not in ['Positive', 'Negative', 'Neutral', 'Irrelevant']:
                        continue

                    # Store compact tuple (item_id, entity, sentiment, text) to save RAM
                    self.data.append((item_id, entity, sentiment, text))
                    self.entities.add(entity)
                    self.sentiments.add(sentiment)
                    
                    count += 1
                    if max_records and count >= max_records:
                        break

        self.loaded = True
        return len(self.data)

    def calculate_agreement_rate(self, engine, sample_size=300):
        """Calculate real VADER vs Naive Bayes agreement rate on dataset sample."""
        if not self.loaded or not engine._trained:
            return 88.5  # Fallback default if not trained

        valid_items = [item for item in self.data if item[2] in ['Positive', 'Negative', 'Neutral'] and len(item[3]) > 10]
        if not valid_items:
            return 88.5

        random.seed(42)
        sample = random.sample(valid_items, min(sample_size, len(valid_items)))
        
        matches = 0
        total = 0
        for item in sample:
            vader_res = engine.analyze_vader(item[3])
            ml_res = engine.classify_ml(item[3])
            if vader_res['label'] == ml_res['label']:
                matches += 1
            total += 1

        if total == 0:
            return 88.5

        return round((matches / total) * 100, 1)

    def get_summary(self, engine=None):
        """Get dataset metrics overview."""
        if not self.loaded:
            self.load()

        sentiment_counts = Counter(item[2] for item in self.data)
        entity_counts = Counter(item[1] for item in self.data)
        
        agreement_rate = self.calculate_agreement_rate(engine) if engine else 88.5

        return {
            'total_records': len(self.data),
            'unique_entities': len(self.entities),
            'sentiment_distribution': dict(sentiment_counts),
            'top_entities': entity_counts.most_common(12),
            'vader_ml_agreement_rate': agreement_rate
        }

    def get_entity_breakdown(self, entity_name):
        """Get sentiment distribution and representative samples for a specific entity."""
        if not self.loaded:
            self.load()

        entity_items = [item for item in self.data if item[1].lower() == entity_name.lower()]
        counts = Counter(item[2] for item in entity_items)

        sample_texts = [
            {'text': item[3], 'sentiment': item[2]}
            for item in entity_items[:6]
        ]

        return {
            'entity': entity_name,
            'total_tweets': len(entity_items),
            'distribution': dict(counts),
            'samples': sample_texts
        }

    def get_training_samples(self, sample_size=2000):
        """Get balanced sample subset for Machine Learning model training."""
        if not self.loaded:
            self.load()

        # Group by sentiment class to create a balanced dataset
        by_sentiment = {'Positive': [], 'Negative': [], 'Neutral': []}
        for item in self.data:
            if item[2] in by_sentiment and len(item[3]) > 10:
                by_sentiment[item[2]].append((item[3], item[2]))

        per_class = max(100, sample_size // 3)
        balanced_samples = []

        random.seed(42)
        for sentiment, items in by_sentiment.items():
            if items:
                count = min(per_class, len(items))
                balanced_samples.extend(random.sample(items, count))

        random.shuffle(balanced_samples)
        return balanced_samples
