import pandas as pd
import numpy as np

import ast
import nltk
import ssl
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import jaccard_score


from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, expr, split
from pyspark.sql.types import FloatType, StringType
from pyspark.sql.functions import col



# Download necessary nltk packages for language processing
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')





# jaccard similarity udf - Strictly for genres
@udf(returnType=FloatType())
def jaccard_similarity_udf(set1_str, set2_str):
    set1 = set(set1_str.split())
    set2 = set(set2_str.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0



class NLPProcessor:

    def __init__(self, sample):

        self.sample = sample


  


    def preprocess_overview(self, text):
        """Will process the overview of a movie and perform
            necessary steps - tokenization and lemmatization"""
        if not isinstance(text, str):
            return []
    
        # tokenize the text
        tokens = word_tokenize(text)

        # Convert to lowercase and remove non-alphanumeric characters
        words = [word.lower() for word in tokens if word.isalnum()]
    
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word not in stop_words]
    
        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    
        return lemmatized_words
    


    def extract_keywords(self, row):
        """Exctract important features and concatenate them into a string
            will be used of text processing"""
        keywords = str(row['keywords'])
        
        
        directors = ' '.join(ast.literal_eval(row['Director']))
        
        overview = row['overview']
        overview_keywords = ' '.join(self.preprocess_overview(overview))

        year = str(row['release_date'])

        cast = row['cast']

        combined_keywords = ' '.join([keywords, year, overview_keywords, directors, cast])
        
        return combined_keywords
    



    def process_text(self, sample):
        """Creates a sample dataframe and processes it using the 
            language processing functions"""
        sample['combined_keywords'] = sample.apply(self.extract_keywords, axis=1)

        sample['genres_set'] = sample.genres.apply(lambda genres: set(ast.literal_eval(genres)))
        sample["genres_set_str"] = sample["genres_set"].apply(lambda x: ' '.join(x))
        sample.drop(['genres_set'], axis=1, inplace=True)
        sample["movie_index"] = sample.index

        return sample



    def process_with_spark(self, sample):
        """Uses Spark to run calculations (jaccard similarity calculation)"""
        self.sample = self.process_text(sample)

        spark = SparkSession.builder.master("local[*]").appName("MovieRecommendation").getOrCreate()
        spark_movies = spark.createDataFrame(sample)


        # Calculate the Jaccard similarity matrix using a Cartesian join
        spark_movies_with_similarity = (
            spark_movies.alias("movies1")
            .crossJoin(spark_movies.alias("movies2"))
            .select(
                col("movies1.movie_index").alias("movie1_index"),
                col("movies2.movie_index").alias("movie2_index"),
                jaccard_similarity_udf("movies1.genres_set_str", "movies2.genres_set_str").alias("jaccard_similarity"),
            )
        )


        pd_movies_with_similarity = spark_movies_with_similarity.toPandas()


        jaccard_similarity_matrix = pd_movies_with_similarity.pivot_table(
            index="movie1_index",
            columns="movie2_index",
            values="jaccard_similarity"
        ).values

        return jaccard_similarity_matrix



    def cosine_matrix(self, sample):
        """Will return a matrix containing the cosine_similarity scores"""

        stop = list(stopwords.words('english'))

        tfidf_vectorizer = TfidfVectorizer(analyzer = 'word', stop_words=list(set(stop)))
        tfidf_matrix = tfidf_vectorizer.fit_transform(sample['combined_keywords'])

        cosine_sim = cosine_similarity(tfidf_matrix)

        return cosine_sim
    

    
    def combine_matrix(self, jaccard_mat, cosine_mat):
        """Combines both jaccard matrix and cosine matrix by assigning
            weights"""
        
        jaccard_weight = 0.25
        cosine_weight = 1 - jaccard_weight

        combined_matrix = (jaccard_weight * jaccard_mat) + (cosine_weight * cosine_mat)

        return combined_matrix
    





    
