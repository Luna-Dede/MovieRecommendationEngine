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

from create_sample import SampleData



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
        self.spark = None
        self.processed_dataframe = None

  


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
        keywords = str(row['keywords'])
        
        
        directors = ' '.join(ast.literal_eval(row['Director']))
        
        overview = row['overview']
        overview_keywords = ' '.join(self.preprocess_overview(overview))

        genres = ' '.join(ast.literal_eval(row['genres'])).lower()

        year = str(row['release_date'])

        cast = row['cast']


        
        combined_keywords = ' '.join([keywords, year, overview_keywords, directors, cast])
        
        return combined_keywords
    



    def process_text(self, sample):
        """Creates a sample dataframe and processes it using the 
            language processing functions"""
        sample_data_obj = SampleData(self.dataframe)
        sample_data_obj.create_sample(sample_size=sample_size)

        sample = sample_data_obj.process_sample()
        sample['combined_keywords'] = sample.apply(self.extract_keywords, axis=1)

        sample['genres_set'] = sample.genres.apply(lambda genres: set(ast.literal_eval(genres)))
        sample["genres_set_str"] = sample["genres_set"].apply(lambda x: ' '.join(x))
        sample.drop(['genres_set'], axis=1, inplace=True)
        sample["movie_index"] = sample.index

        return sample



    def process_with_spark(self):
        processed_df = self.process_dataframe(sample_size)

        spark = SparkSession.builder.master("local[*]").appName("MovieRecommendation").getOrCreate()
        spark_movies = spark.createDataFrame(processed_df)


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



    def similarity(self):
        pass


df = pd.read_csv("data/final_movie.csv")
nlp_proc = NLPProcessor(df)

print(nlp_proc.process_with_spark())

        
    



    
