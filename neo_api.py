from neo4j import GraphDatabase
from TextProcessing.nlp_processing import *
from TextProcessing.create_sample import SampleData




class Neo4jAPI:
    """Neo4jAPI class that will do further data preparation and set up the
        graph database"""

    def __init__(self, dataframe, uri, user, password):
        self.dataframe = dataframe
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    
    def close(self):
        self.driver.close()


    def create_sample(self, sample_size=2000):
        sample_data_obj = SampleData(self.dataframe)
        sample_data_obj.create_sample(sample_size=sample_size)

        self.sample = sample_data_obj.process_sample()



    def generate_data(self):
        
        # Instantiate nlp object and create processed sample
        nlp_obj = NLPProcessor(self.sample)

        # Compute similarity matrices
        jaccard_matrix = nlp_obj.process_with_spark(self.sample)
        cosine_matrix = nlp_obj.cosine_matrix(self.sample)

        # Combine matrices
        combined_matrix = nlp_obj.combine_matrix(jaccard_matrix, cosine_matrix)
        sample = nlp_obj.sample


        # create matrix into dataframe
        mat = np.matrix(combined_matrix)
        mat_df = pd.DataFrame(data=mat.astype(float))


        mat_df.to_csv('data/scores_matrix.csv', sep=' ', header=True, index=False)
        sample.to_csv("data/sample.csv")


        
    def create_nodes_and_relationships(self, tx, movie_title, genres, directors, overview, vote_count, vote_average, movie_index):
        
        """Creates the nodes and the relationships between each node
            Director node
            Genre node
            Movie node"""
        

        tx.run("""
            MERGE (m:Movie {title: $movie_title})
            SET m.overview = $overview,
                m.vote_count = $vote_count,
                m.vote_average = $vote_average
            """, movie_title=movie_title, overview=overview, vote_count=vote_count, vote_average=vote_average)

        for genre in genres:
            tx.run("""
                MATCH (m:Movie {title: $movie_title})
                MERGE (g:Genre {name: $genre})
                MERGE (m)-[:HAS_GENRE]->(g)
                """, movie_title=movie_title, genre=genre)


        for director in directors:
            tx.run("""
                MATCH (m:Movie {title: $movie_title})
                MERGE (d:Director {name: $director})
                MERGE (m)-[:DIRECTED_BY]->(d)
                """, movie_title=movie_title, director=director)
            



    def process_dataframe_and_create_nodes(self, df):

        """Process dataframe and pass columns as parameters for the 
            create_nodes_and_relationships method"""
        with self.driver.session() as session:
            for index, row in df.iterrows():
                movie_title = row['title']
                genres = ast.literal_eval(row['genres'])
                directors = ast.literal_eval(row['Director'])
                overview = row['overview']
                vote_count = row['vote_count']
                vote_average = row['vote_average']
                movie_id = row['movie_index']

                session.write_transaction(self.create_nodes_and_relationships, movie_title, genres, directors, overview, vote_count, vote_average, movie_id)
            
        


    # create similarity relationship between two movies
    @staticmethod
    def _create_relationships_for_sample(tx, sample, similarity_matrix, threshold):
        """Will create relationship between two movies based on 
            similarity if the cosine score is greater than the threshold"""
        for i in range(len(sample)):
            for j in range(i+1, len(sample)):
                distance = similarity_matrix[i][j]
                if i != j and distance > threshold:
                    movie1_title = sample.iloc[i]['title']
                    movie2_title = sample.iloc[j]['title']
                    tx.run("""
                        MATCH (m1:Movie {title: $movie1_title}), (m2:Movie {title: $movie2_title})
                        CREATE (m1)-[:SIMILAR {distance: $distance}]->(m2)
                    """, movie1_title=movie1_title, movie2_title=movie2_title, distance=distance)


    def create_relationships(self, sample, similarity_matrix, threshold):
        with self.driver.session() as session:
            session.write_transaction(self._create_relationships_for_sample, sample, similarity_matrix, threshold)




    # Flush all nodes and properties in the graph
    @staticmethod
    def _delete_all_nodes_and_relationships(tx):
        tx.run("MATCH (n) DETACH DELETE n")


    def flush(self):
        with self.driver.session() as session:
            session.write_transaction(self._delete_all_nodes_and_relationships)






