import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd 
import numpy as np

def tf_idf(df):
    """
    This function performs Term Frequency-Inverse Document Frequency (TF-IDF)
    analysis on a DataFrame of text reviews and returns a dictionary of terms
    and their summed TF-IDF scores.

    Parameters:
    df (pandas.DataFrame): The DataFrame of text reviews.

    Returns:
    dict: A dictionary where the keys are terms and the values are the summed
          TF-IDF scores for those terms across all documents.

    Raises:
    ValueError: If the input DataFrame is empty.
    """

    # Drop any rows in the DataFrame that contain missing values
    review = df.dropna()

    # Raise an exception if the DataFrame is empty
    if review.empty:
        raise ValueError("Input DataFrame is empty.")

    # Extract the values as a NumPy array
    review = review.values

    # Initialize a TF-IDF vectorizer with the option to remove stop words
    tfidf_vectorizer = TfidfVectorizer()

    # Fit the TF-IDF vectorizer to the text data and transform it into a
    # sparse matrix representation
    tfidf_matrix = tfidf_vectorizer.fit_transform(review)

    # Sum the TF-IDF scores across all documents for each term directly on
    # the sparse matrix. This produces a 1D array of summed TF-IDF scores.
    tfidf_sum = np.array(tfidf_matrix.sum(axis=0)).flatten()

    # Get the feature names (words) that correspond to the columns of the
    # TF-IDF matrix
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Create a DataFrame where the rows are the terms and the columns are the
    # summed TF-IDF scores for those terms across all documents.
    tfidf_sum_df = pd.DataFrame({'term': feature_names, 'tfidf_sum': tfidf_sum})

    # Sort the DataFrame by the summed TF-IDF scores in descending order
    tfidf_sum_df = tfidf_sum_df.sort_values(by='tfidf_sum', ascending=False)

    # Convert the sorted DataFrame into a dictionary where the keys are the
    # terms and the values are the summed TF-IDF scores for those terms.
    tfidf_dict = dict(zip(tfidf_sum_df['term'], tfidf_sum_df['tfidf_sum']))

    # Return the dictionary of terms and their summed TF-IDF scores
    return tfidf_dict
