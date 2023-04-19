import pandas as pd




class SampleData:

    def __init__(self, data) -> None:
        self.data = data


    

    def create_sample(self, sample_size=2000, random_state=None):
        """Will create a sample dataset given  """
        self.sample = self.data.sample(sample_size, random_state=random_state)
    


    def process_sample(self):
        """Process sample"""
        sample = self.sample

        # Replace missing values with strings
        sample['cast'].fillna("", inplace=True)
        sample['overview'].fillna("", inplace=True)
        sample['keywords'].fillna("", inplace=True)

        sample.reset_index(inplace=True)
        sample.drop(['index'], axis=1, inplace=True)

        return sample
    



