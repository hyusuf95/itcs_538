from Housing import Housing
from pandas.core.frame import DataFrame

house: Housing = Housing('data.csv')



# trained_model: DataFrame = house.train_data()
# #print the type of trained_model
# print(trained_model.head())



house.correlation_coefficients()