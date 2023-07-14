from Housing import Housing
from pandas.core.frame import DataFrame

house: Housing = Housing('data.csv')



prepared_data = house.prepare_data()


trained_model: DataFrame = house.tain_data()
#print the type of trained_model
(trained_model.describe())