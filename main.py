from neo_api import Neo4jAPI
import pandas as pd
import numpy as np



uri = "neo4j://localhost:7687"
user = "neo4j"
password = "vampire-float-olivia-maestro-sleep-4222" 


# reads in the full dataset
dataset = pd.read_csv("data/final_movie.csv")



def main():

    api_object = Neo4jAPI(dataset, uri, user, password) # create api object

    # flush all existing nodes
    api_object.flush()
    # will create a random sample
    api_object.create_sample()

    # Will create the sample and similarity datasets
    api_object.generate_data()


    sample = pd.read_csv("data/sample.csv")
    scores_matrix = pd.read_csv("data/scores_matrix.csv", sep=" ")


    # convert scores matrix to numpy array
    rec_scores = scores_matrix.to_numpy()
    # compute threshold with non-zero values (only scores that show significance in similarity)
    non_zero_scores = rec_scores[rec_scores > 0]
    mean_score = np.mean(non_zero_scores)
    std_score = np.std(non_zero_scores)

    # Set the threshold as the mean plus some multiple of the standard deviation
    threshold = mean_score + 1 * std_score


    # create nodes
    api_object.process_dataframe_and_create_nodes(sample)
    api_object.create_relationships(sample, rec_scores, threshold)






if __name__ == "__main__":
    main()


