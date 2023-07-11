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
        return self.housing.isnull().sum()*100/self.housing.shape[0]



# # Outlier Analysis
# fig, axs = plt.subplots(2,3, figsize = (10,5))
# plt1 = sns.boxplot(housing['price'], ax = axs[0,0])
# plt2 = sns.boxplot(housing['area'], ax = axs[0,1])
# plt3 = sns.boxplot(housing['bedrooms'], ax = axs[0,2])
# plt1 = sns.boxplot(housing['bathrooms'], ax = axs[1,0])
# plt2 = sns.boxplot(housing['stories'], ax = axs[1,1])
# plt3 = sns.boxplot(housing['parking'], ax = axs[1,2])

# plt.tight_layout()

    def outlier_analysis(self):
        fig, axs = plt.subplots(2,3, figsize = (10,5))
        plt1 = sns.boxplot(self.housing['price'], ax = axs[0,0])
        plt2 = sns.boxplot(self.housing['area'], ax = axs[0,1])
        plt3 = sns.boxplot(self.housing['bedrooms'], ax = axs[0,2])
        plt1 = sns.boxplot(self.housing['bathrooms'], ax = axs[1,0])
        plt2 = sns.boxplot(self.housing['stories'], ax = axs[1,1])
        plt3 = sns.boxplot(self.housing['parking'], ax = axs[1,2])

        plt.tight_layout()
        plt.show()

        
         




