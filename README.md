# Finding Optimal Number of Clusters in K-means Algorithm

The K-means algorithm is a popular clustering technique that “hard” partitions data
into K clusters. However, determining the optimal number of clusters K is a crucial yet challenging task. This project explores the application of **Occam’s Razor** to guide the selection of the optimal K value in **K-means clustering**.

The pipeline involves robust text pre-processing, transforming the text into a word co-occurrence matrix (a form of distributional word embedding), and employing K-Fold Cross-Validation to rigorously tune and evaluate the clustering model. The optimal K is then determined by analyzing two primary evaluation metrics: Inertia and Silhouette Score.

## Dataset
The analysis is performed on the publicly available *text8* corpus, a widely used dataset for text modeling and word embedding experiments.

**Download Link:** [http://mattmahoney.net/dc/text8.zip](http://mattmahoney.net/dc/text8.zip)

**Setup:** Please download and unzip the file, then update the *file_path* variable in the ```word_clustering.py``` script to point to the local location of the text8 file.

## Installation

Make sure you have Python installed on your system. You can verify your installation by running:
```bash
python --version
```
Install all the required Python dependencies:
```bash
pip install -r requirements.txt
```

## Run the Code
```bash
python Kmean_clustering.py
```
