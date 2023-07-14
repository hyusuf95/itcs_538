import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

class Housing:
    def __init__(self, data_path):
        self.data_path = data_path
        self.housing = pd.read_csv(self.data_path)
    
    def get_head(self, n=5):
        return self.housing.head(n)


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
    
    def describe(self):
        return self.housing.describe()


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




    def visualising_numeric_variables(self):
        sns.pairplot(self.housing)
        plt.show()


    def visualising_categorical_variables(self):
        plt.figure(figsize=(20, 12))
        plt.subplot(2,3,1)
        sns.boxplot(x = 'mainroad', y = 'price', data = self.housing)
        plt.subplot(2,3,2)
        sns.boxplot(x = 'guestroom', y = 'price', data = self.housing)
        plt.subplot(2,3,3)
        sns.boxplot(x = 'basement', y = 'price', data = self.housing)
        plt.subplot(2,3,4)
        sns.boxplot(x = 'hotwaterheating', y = 'price', data = self.housing)
        plt.subplot(2,3,5)
        sns.boxplot(x = 'airconditioning', y = 'price', data = self.housing)
        plt.subplot(2,3,6)
        sns.boxplot(x = 'furnishingstatus', y = 'price', data = self.housing)
        plt.show()



    def convert_to_binaries(self):
        #Convert 'Yes and No' to 1 and 0
        cat_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
        self.housing[cat_cols] = self.housing[cat_cols].apply(convert_to_binary)
        return self.housing
    

    def dummy_variables(self):
        #Dummy Variables for furnishingstatus column
        #Append the new columns to the dataframe
        status = pd.get_dummies(self.housing['furnishingstatus'])
        self.housing = pd.concat([self.housing, status], axis = 1)
        self.housing.drop(['furnishingstatus'], axis = 1, inplace = True)
        return self.housing
    

    # def reduce_redundunt_columns(self):
    #     #Normalize the data
    #     status = pd.get_dummies(self.housing['furnishingstatus'], drop_first = True)
    #     housing = pd.concat([self.housing, status], axis = 1)
    #     return housing




    def prepare_data(self):
        self.convert_to_binaries()
        self.dummy_variables()
        #self.reduce_redundunt_columns()
        return self.housing




    def tain_data(self):
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.model_selection import train_test_split



        #Prepare data

        print(self.prepare_data())
        


        #Split data to train and test
        np.random.seed(0)
        df_train, df_test = train_test_split(self.housing, train_size = 0.7, test_size = 0.3, random_state = 100)
        scaler = MinMaxScaler()
        num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price']
        df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
        return df_train












def convert_to_binary(x):
    return x.map({'yes': 1, "no": 0})
        
        
         




