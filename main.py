from neo_api import Neo4jAPI
import pandas as pd



uri = "neo4j://localhost:7687"
user = "neo4j"
password = "vampire-float-olivia-maestro-sleep-4222" 



dataset = pd.read_csv("data/final_movie.csv")



def main():

    api_object = Neo4jAPI(dataset, uri, user, password) # create api object

    api_object.create_sample()






if __name__ == "__main__":
    pass


