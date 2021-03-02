import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

header  = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
modelTraining= st.beta_container()

@st.cache
def get_data(filename):
	taxi_data = pd.read_csv(filename)
	return taxi_data

with header:
	st.title('Welcome to my awesome data science project!')
	st.text('in this project I look into the transactions of taxis in NYC')

with dataset:
	st.header('NYC taxi dataset')
	st.text('I found this dataset on kaggle.blah blah blah.com')


	taxi_data = get_data('Data/taxi_data.csv')
	st.write(taxi_data.head())
	st.subheader('Passenger Count Distribution on NYC Dataset')
	pulocation_distribution = pd.DataFrame(taxi_data['passenger_count'].value_counts())
	st.bar_chart(pulocation_distribution)

with features:
	st.header('The features I created')
	st.markdown('')
features.slider('DBfeaturesA',0,20,10)



with modelTraining:
    st.header('Model training')
    st.text('In this section you can select the hyperparameters!')

    selection_col, display_col = st.beta_columns(2)

    max_depth = selection_col.slider('What should be the max_depth of the model?', min_value=10, max_value=100, value=20, step=10)

    number_of_trees = selection_col.selectbox('How many trees should there be?', options=[100,200,300,'No limit'],index=0)

    selection_col.text('Here is a list of features: ')
    selection_col.write(taxi_data.columns)
    input_feature = selection_col.text_input('Which feature would you like to input to the model?', 'PULocationID')



# regr = RandomForestRegressor(max_depth=max_depth,n_estimators=number_of_trees)    #1

# X = taxi_data[[input_feature]]     #2
# y = taxi_data[[‘trip_distance’]]     #3

# regr.fit(X, y) #4
# prediction = regr.predict(y) #5

# display_col.subheader('Mean squared error:')
# display_col.write(mean_squared_error(y, prediction))

# display_col.subheader('Mean absolute error:')
# display_col.write(mean_absolute_error(y, prediction))

# display_col.subheader('R squared score of the model is:')
# display_col.write(mean_absolute_error(y, prediction))

#    option = st.selectbox('How would you like to be contacted?',('Email', 'Home phone', 'Mobile phone'))