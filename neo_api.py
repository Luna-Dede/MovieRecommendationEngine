from neo4j import GraphDatabase
from TextProcessing.create_sample import SampleData
from TextProcessing.nlp_processing import *



class Neo4jAPI:

    def __init__(self, dataframe, uri, user, password):
        self.dataframe = dataframe
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

       # self.delete = pass

    def create_sample(self, sample_size=2000):
        sample_data_obj = SampleData(self.dataframe)
        sample_data_obj.create_sample(sample_size=sample_size)

        self.sample = sample_data_obj.process_sample()



    def process_sample(self):
        pass



    def create_movie_node(self):
        pass


    def create_genre_node(self):


        pass


    def flush(self, tx):

        tx.run("MATCH (n) DETACH DELETE n")

        pass