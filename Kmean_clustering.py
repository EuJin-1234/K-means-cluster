from urllib import request
import nltk, re, pprint
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class WordClusteringPipeline:
    """
    A pipeline for pre-processing text, building a co-occurrence matrix,
    and performing K-Means clustering with cross-validation.
    """
    def __init__(self, file_path, threshold=50, window_size=2, test_size=0.2, val_size=0.25, random_state=42):
        self.file_path = file_path
        self.threshold = threshold
        self.window_size = window_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.k_values = range(2, 11)
        self.folds = 5
        self.words = None

    def _preprocess_text(self):
        """
        Performs text pre-processing steps: read, clean, tokenize, filter,
        remove stopwords, lemmatize, and frequency filter.
        """
        print("--- Text Pre-processing ---")
        with open(self.file_path, 'r', encoding='utf8') as file:
            raw = file.read()

        # Clean non-ASCII characters
        raw_ascii = raw.encode("ascii", "ignore").decode("ascii")

        # Tokenize and remove punctuation
        tokenizer = RegexpTokenizer(r'\w+')    
        tokens = tokenizer.tokenize(raw_ascii)
        print(f"Number of tokens after initial tokenization: {len(tokens)}")

        # Lowercase and filter by length
        filtered_tokens = [w.lower() for w in tokens if 3 <= len(w) < 18]
        print(f"Number of tokens after length filtering: {len(filtered_tokens)}")

        # Load and update stopwords
        stop_words = set(stopwords.words('english'))
        custom_stopwords = {'also', 'may', 'many', 'would', 'however'}
        stop_words.update(custom_stopwords)

        # Remove stopwords
        words = [word for word in filtered_tokens if word not in stop_words]
        print(f"Number of tokens after removing stopwords: {len(words)}")

        # Apply Lemmatization
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        # Remove number words
        number_words = {'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth'}
        words = [word for word in words if word not in number_words]

        # Frequency filtering
        word_freq = Counter(words)
        self.words = [word for word in words if word_freq[word] >= self.threshold]
        
        # Display 10 most common words before final filter
        most_common_words = Counter(words).most_common(10)
        print("\nTop 10 most common words (pre-final filter):")
        for word, count in most_common_words:
             print(f"{word}: {count}")

        print(f"\nNumber of tokens after frequency filtering (threshold={self.threshold}): {len(self.words)}")
        print(f"Total number of unique words (vocabulary size): {len(set(self.words))}")

    def _normalize_matrix(self, matrix):
        """Applies Min-Max normalization to the co-occurrence matrix."""
        scaler = MinMaxScaler()
        return scaler.fit_transform(matrix)

    def _build_co_matrix(self, words_subset, unique_words_vocab, window_size):
        """
        Builds the co-occurrence matrix for a subset of words based on a given vocabulary.
        """
        word_index = {word: idx for idx, word in enumerate(unique_words_vocab)}
        co_occurrences = defaultdict(Counter)

        for i, word in enumerate(words_subset):
            if word in word_index:
                for j in range(max(0, i - window_size), min(len(words_subset), i + window_size + 1)):
                    if i != j and words_subset[j] in word_index:
                        co_occurrences[word][words_subset[j]] += 1

        # Initialize matrix
        co_matrix = np.zeros((len(unique_words_vocab), len(unique_words_vocab)), dtype=int)
        
        # Populate the matrix
        for word, neighbors in co_occurrences.items():
            for neighbor, count in neighbors.items():
                co_matrix[word_index[word]][word_index[neighbor]] = count

        return self._normalize_matrix(co_matrix)

    def run_pipeline(self):
        """Executes the full clustering pipeline, including train/val/test split and cross-validation."""
        if self.words is None:
            self._preprocess_text()

        words_array = np.array(self.words)

        # 1. Split into (Train + Validation) and Test sets
        train_val_words, test_words = train_test_split(
            words_array, test_size=self.test_size, random_state=self.random_state
        )
        print(f"\nTotal words: {len(words_array)}")
        print(f"Train + Validation set size: {len(train_val_words)}")
        print(f"Test set size: {len(test_words)}")

        train_inertias = {k: [] for k in self.k_values}
        train_silhouette_scores = {k: [] for k in self.k_values}
        val_silhouette_scores = {k: [] for k in self.k_values}
        
        print("\n--- Starting K-Fold Cross-Validation ---")
        
        # 2. Perform k-fold cross-validation on the Train + Validation set
        kf = KFold(n_splits=self.folds, shuffle=True, random_state=self.random_state)

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_words)):
            train_fold_words, val_fold_words = train_val_words[train_idx], train_val_words[val_idx]
            
            unique_train_vocab = list(set(train_fold_words))
            
            train_co_matrix = self._build_co_matrix(train_fold_words, unique_train_vocab, self.window_size)
            val_co_matrix = self._build_co_matrix(val_fold_words, unique_train_vocab, self.window_size)

            for k in self.k_values:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init='auto')
                kmeans.fit(train_co_matrix)

                # Evaluate on Training Fold
                train_labels = kmeans.labels_
                train_inertias[k].append(kmeans.inertia_)
                if len(set(train_labels)) > 1:
                    train_silhouette_scores[k].append(silhouette_score(train_co_matrix, train_labels))
                else:
                    train_silhouette_scores[k].append(0) # or -1 as a placeholder

                # Evaluate on Validation Fold
                val_labels = kmeans.predict(val_co_matrix)
                if len(set(val_labels)) > 1 and len(val_labels) > k:
                    val_silhouette_scores[k].append(silhouette_score(val_co_matrix, val_labels))
                else:
                    val_silhouette_scores[k].append(0) 
            
            print(f"Fold {fold+1}/{self.folds} processed.")


        # 3. Average results across folds
        avg_train_inertias = [np.mean(train_inertias[k]) for k in self.k_values]
        avg_train_silhouette_scores = [np.mean(train_silhouette_scores[k]) for k in self.k_values]
        avg_val_silhouette_scores = [np.mean(val_silhouette_scores[k]) for k in self.k_values]
        
        # Determine the best K from validation scores
        best_k_index = np.argmax(avg_val_silhouette_scores)
        best_k = self.k_values[best_k_index]
        print(f"\nBest K suggested by average validation silhouette score: {best_k}")

        # Build final model on the entire Train + Validation set
        final_vocab = list(set(train_val_words))
        train_val_co_matrix = self._build_co_matrix(train_val_words, final_vocab, self.window_size)
        test_co_matrix = self._build_co_matrix(test_words, final_vocab, self.window_size)

        final_kmeans = KMeans(n_clusters=best_k, random_state=self.random_state, n_init='auto')
        final_kmeans.fit(train_val_co_matrix)
        
        # Evaluate on Test Set
        test_labels = final_kmeans.predict(test_co_matrix)
        if len(set(test_labels)) > 1 and len(test_labels) > best_k:
            test_silhouette = silhouette_score(test_co_matrix, test_labels)
            print(f"Final Test Set Silhouette Score (K={best_k}): {test_silhouette:.4f}")
        else:
            print("Cannot calculate Test Set Silhouette Score (too few unique labels or data points).")
            
        self._plot_results(avg_train_inertias, avg_train_silhouette_scores, avg_val_silhouette_scores)

    def _plot_results(self, avg_train_inertias, avg_train_silhouette_scores, avg_val_silhouette_scores):
        """Generates plots for the Elbow method and Silhouette Scores."""
        plt.figure(figsize=(15, 6))

        # Elbow Plot
        plt.subplot(1, 2, 1)
        plt.plot(self.k_values, avg_train_inertias, marker='o', label='Average Train Inertia')
        plt.title("Elbow Method (K-Fold Cross-Validation)")
        plt.xlabel("Number of clusters (K)")
        plt.ylabel("Inertia (Sum of Squared Distances)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        # Silhouette Score Plot
        plt.subplot(1, 2, 2)
        plt.plot(self.k_values, avg_train_silhouette_scores, marker='o', label='Average Train Silhouette Score')
        plt.plot(self.k_values, avg_val_silhouette_scores, marker='x', label='Average Validation Silhouette Score', linestyle='--')
        plt.title("Silhouette Score (Train vs Validation)")
        plt.xlabel("Number of clusters (K)")
        plt.ylabel("Silhouette Score")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # Replace this with the path where you store the text8 file.
    file_path = r"data/Your_Dataset_Path" 

    pipeline = WordClusteringPipeline(file_path=file_path)
    pipeline.run_pipeline()