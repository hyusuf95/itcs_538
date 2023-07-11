import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

class Housing:
    def __init__(self, data_path):
        self.data_path = data_path
        self.housing = pd.read_csv(self.data_path)
    
    def get_head(self):
        return self.housing.head()


    def get_info(self):
        return self.housing.info()

    def get_describe(self):
        return self.housing.describe()

    def get_columns(self):
        return self.housing.columns


    def inspect(self):
        return self.housing.shape


    def clean(self):
        self.housing.isnull().sum()*100/self.housing.shape[0]
        return self.housing.shape




