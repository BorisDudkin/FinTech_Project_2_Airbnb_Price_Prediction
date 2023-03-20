import pickle
import time
from pathlib import Path

import chart_studio.plotly as py
import cufflinks as cf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, model_from_json

st.set_page_config(page_title="AirBnB Price Prediction with ML", layout='wide')

@st.cache_data
def get_data(file_path):
    df = pd.read_csv(file_path)
    return df

file_path = Path("./Resources/air_bnb.csv")
df_original = get_data(file_path)
bnb_df=df_original.copy()
bnb_df_map=bnb_df.loc[bnb_df['realSum']<=500]

with st.sidebar:
    st.subheader('Data Analysis')
    city_selection = st.selectbox("Select a city for data analysis (when required):", tuple(bnb_df['city'].unique()))


    st.subheader('Machine Learning')
    my_mlm=['Random Forest','Neural Network']
    mlm_selection = st.selectbox("Select a Nachine Learning Model:", tuple(my_mlm))
    if mlm_selection== 'Random Forest':
        max_depth = st.slider("Depth of the Model:", min_value=10, max_value=100, value=20, step = 10)
        n_estimators=st.selectbox("Number of Estimators:",options=[100, 150, 200, 250, 300], index = 0)
    else:
        n_epochs= st.slider("Number of Epochs:", min_value=50, max_value=500, value=100, step = 50)
        n_layers=st.selectbox("Number of Hidden Layers:",options=[1, 2, 3], index = 0)

# Create four tabs to display our application as per below breakdown
tab1, tab2, tab3, tab4, tab5 = st.tabs(['Introduction','Data Analysis and Cleansing','Exploratory Data Analysis','Machine Learning', 'Model Prediction'])

with tab1:

    icon,title=st.columns([1,20])
    with icon: 
        st.image('Images/bnb.png', use_column_width=True)
    with title:
        st.title('Airbnb Price Prediction (European cities)')

    st.markdown('**Please select a a :blue[city] in the sidebar to explore Airbnb interactive map by location.**')

    fig_map = px.density_mapbox(bnb_df_map.loc[bnb_df_map['city']==city_selection], lat = 'lat', lon = 'lng', z = 'realSum', radius = 10, center = dict(lat=bnb_df_map.loc[bnb_df['city']==city_selection]['lat'].mean(), lon = bnb_df_map.loc[bnb_df['city']==city_selection]['lng'].mean()),
                                zoom = 11, hover_name='room_type', mapbox_style = 'open-street-map',
                            title = 'Explore Listings in '+city_selection,labels={'realSum': 'Listing Price' })
    st.plotly_chart(fig_map,use_container_width=True)

    st.subheader('Project Description:')

    st.markdown('''
        The project consists of following parts:
* Initial data analysis - reviewing data, providing more descriptive names to columns, removing redundant columns, checking for descriptive statistics, missing values, duplicates, data distribution and  outliers and correcting for all of those if needed.
* Exploratory data analysis - analyzing relationships between the features and a target (listing prices).
* Applying Machine Learning regression models to predict the Airbnb properties' listings prices based on a set of selected features; calibrating the models by changing features and model parameters to optimize the performance; selecting a better performing model.
* Applying the saved model to a user input to determine a property listing price.
''')

    st.subheader('Project Objective:')

    st.markdown('''The objective of the project is two-fold:
1. Help Airbnb owners understanding the relationship between different characteristics of their property to potentially improve services and profitability of their business.
2. Apply machine learning to the data set to be able to predict the listing price by providing selected features (based on the model optimization analysis). The ultimate objective here ranges from assisting:
   - a traveler to estimate the Airbnb price in the vacation destination'
   - new business owners who are trying to set an appropriate competitive price for their property;
   - new and existing business owners who attempt to estimate what property characteristics affect the price of their property and how they can potentially improve their profitability.''')

    st.subheader('Dataset:')

    st.markdown('''[Airbnb Price Determinants in Europe](https://www.kaggle.com/datasets/thedevastator/airbnb-price-determinants-in-europe) dataset can be found on Kaggle. The website also provides a descriptions of the data characteristics, possible dataset use cases and potential research ideas''')

    st.subheader('Other Areas to Investigate:')

    st.markdown('''As the dataset contains cystomer satisfuction information, the next step could be to utilize Machine Learning Classifiers to study the relationship between the variables in the dataset and the custormer satisfaction. Such an alaysis could be useful for Airbnb owners who aim at optimizing different characteristics of their property to achive the maximum client satisfaction ''')

with tab2:

    icon,title=st.columns([1,20])
    with icon: 
        st.image('Images/home.png', use_column_width=True)
    with title:
        st.title('Initial Data Analysis')

    with st.expander("About this section"):
        st.markdown("Initial Data Analysis begins with demonstrating the original dataset. It then continues by showing the progress in data cleansing - removing redundant features, giving columns more descriptive names, analizing the descriptive statistics of the data, its shape, checking for missing values duplicates and the types of data prsent in the dataset. The section continues with univariate analysis of the features and focuses on the outliers for our target variable _lisitng price_. The correction for outliers is suggested and the dataset after the correction is demostrated breaking the information by city.")

    st.header('Original Data Review and Cleansing:')

    st.subheader('Original Dataset (top 5 rows) :')
    st.dataframe(df_original.head(),use_container_width=True)

    # remove redundant columns
    bnb_df=bnb_df.drop(columns=['room_shared','room_private','attr_index','attr_index_norm','rest_index','rest_index_norm', 'lng', 'lat'])

    # bnb_df=bnb_df.drop(columns=['room_shared','room_private','attr_index','attr_index_norm','rest_index','rest_index_norm'])    

    # rename columns with more descriptive names:
    bnb_df.rename(columns = {'day':'day_of_week','realSum':'listing_price', 'multi':'multiple_rooms ',
                              'biz':' business_facilities', 'bedrooms':'bedrooms_quantity', 'dist':'city_center_distance', 'metro_dist':'metro_distance'}, inplace = True)

    #modify columns with Boolean data:
    bnb_df['host_is_superhost']=bnb_df['host_is_superhost'].apply(lambda x:1 if x==True else 0)

    # use bnb_df to get user input characteristics (features) for listing price prediction:
    feater_columns = bnb_df.drop(columns=['listing_price']).columns.to_list()

    categorical = list(bnb_df.dtypes[bnb_df.dtypes == "object"].index)
    # create empty dictionary to store the user input   
    input_dict={}
    st.sidebar.subheader('Airbnb Charcteristics:')
    for column in feater_columns:
        if column == "guest_satisfaction_overall":
            input_dict[column] = st.sidebar.slider(column, min_value=20, max_value=100, value=90, step = 1)
        elif column == "metro_distance":
            input_dict[column] = st.sidebar.slider(column, min_value=0.01, max_value=8.00, value=1.01, step = 0.01)
        elif column =="city_center_distance":
            input_dict[column] = st.sidebar.slider(column, min_value=0.01, max_value=15.00, value=1.01, step = 0.01)
        elif column in categorical:
            user_input =st.sidebar.selectbox(column, tuple(bnb_df[column].sort_values().unique()))
            for char in  bnb_df[column].unique():
                if char ==user_input:
                    input_dict[column+'_'+char] = 1
                else:
                    input_dict[column+'_'+char] = 0 
        else:
            input_dict[column] =float(st.sidebar.selectbox(column, tuple(bnb_df[column].sort_values().unique())))
   
    # create a dataframe based on the user inlut about airbnb listing
    input_df=pd.DataFrame(data = input_dict, index=[0])

    st.subheader('Modified Dataset (top 5 rows)')
    st.caption('NOTE: Redundant columns removed, descriptive names given to columns, boolean type columns modified.')
    st.dataframe(bnb_df.head(),use_container_width=True)

    st.subheader('Descriptive Statistics:')
    st.write(bnb_df.describe(),use_container_width=True)


    st.subheader('Dataset Deepdive:')
    col1,col2,col3,col4 = st.columns(4, gap='large')
    
    with col1:
        st.markdown('#### Dataset Shape:')
        st.write(bnb_df.shape,use_container_width=True)
        st.caption('The modified dataset has 51,707 rows and 13 columns.')
    with col2:
        st.markdown(' #### Duplicated Rows:')
        st.write(bnb_df.duplicated().sum(),use_container_width=True)
        st.caption('There are no duplicated rows.')
    with col3:
        st.markdown(' #### Misssing Values:')
        st.write(bnb_df.isnull().sum(),use_container_width=True)
        st.caption('There are no missing values.')
    with col4:
        st.markdown(' #### Columns Datatypes:')
        st.write(bnb_df.dtypes,use_container_width=True)
        st.caption('There are three columns with cathegorical data: day_of_week, room_type and city, which is in line with expectations.')
    
    st.subheader('Data Distirbution and Outliers:')

    column_selection = st.selectbox("Select a Dataframe column to analyze the number of unique values and distribution:", tuple(bnb_df.columns))
    
    col1,col2 = st.columns(2, gap='large')
  
    with col1:
        # st.write(bnb_df[column_selection].value_counts())
        fig = px.bar(bnb_df[column_selection].value_counts(), y=column_selection, labels={'index': 'Unique Values' }, title = column_selection + ' - Unique Values Count')
        fig.update_layout(uniformtext_minsize=8, yaxis_title='Unique Values count', xaxis_title=column_selection)
        st.plotly_chart(fig,use_container_width=True)

    with col2:
        fig_hist=px.histogram(bnb_df, x=column_selection, nbins=11, title=column_selection + ' Histogram', marginal = 'violin')
        st.plotly_chart(fig_hist,use_container_width=True)

    st.subheader('Listing Prices Outliers:')
    st.caption('Please select a a **:blue[city]** on the sidebar.')
    # city_selection = st.selectbox("Select a city to analyse further the listing price outliers:", tuple(bnb_df['city'].unique()))

    col1,col2 = st.columns(2, gap='large')
    with col1:

        fig_quantile=px.bar(bnb_df.loc[bnb_df['city']==city_selection]['listing_price'].quantile([0.01, 0.25, 0.5, 0.75, 0.99]),labels={'index': 'Quantile', 'value': 'Listing Price' }, title = city_selection + ' - Listing Price Quantile Distribution')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig_quantile,use_container_width=True)

        st.caption('Based on the descriptive statistics of the Dataframe and on the distribution analysis the lisiting prices above **:blue[500]** can be considered as outliers and will be removed for the Exploratory Data Analysis.')

    with col2:

        fig_city=go.Figure()
        fig_city.add_trace(go.Violin(x=bnb_df['city'][bnb_df['city']==city_selection],
                       y=bnb_df['listing_price'][bnb_df['day_of_week']=='weekday'],
                       legendgroup='Yes', scalegroup='Yes', name='weekday',
                       side='negative', line_color='blue'))
        fig_city.add_trace(go.Violin(x=bnb_df['city'][bnb_df['city']==city_selection],
                       y=bnb_df['listing_price'][bnb_df['day_of_week']=='weekend'],
                       legendgroup='Yes', scalegroup='Yes', name='weekend',
                       side='positive', line_color='red'))
        fig_city.update_layout(title=city_selection +' City Violin Plot', yaxis_title='Listing Price')
        st.plotly_chart(fig_city,use_container_width=True)
        st.markdown(f' ##### {city_selection} Listing Price Quantiles')
    
    # remove outliers (listing prices above 650):
    df=bnb_df.loc[bnb_df['listing_price']<=500]
    df = df.reset_index(drop=True)
    st.markdown('#### Dataset Shape corrected for outliers:')
    st.write(df.shape,use_container_width=True)
    # st.dataframe(df.tail(),use_container_width=True)
    price_outliers_new=px.violin(df, x='city', y='listing_price', box=True, 
                color='day_of_week', points='all', hover_data=df.columns,
                labels={'city': 'City','listing_price': 'Listing Price' }, title = 'Listing Price Distribution per city (corrected for outliers)')
    st.plotly_chart(price_outliers_new,use_container_width=True)
   
    
with tab3:

    icon,title=st.columns([1,20])
    with icon: 
        st.image('Images/rent.png', use_column_width=True)
    with title:
        st.title('Exploratory Data Analysis')

    with st.expander("About this section"):
        st.markdown("Exploratory Data Analysis continues checking the data but this time focusing on the bivariate and multivariate relationship between the features and the target as well as between different features, including categorical/numerical and numerical/numerical analysis. The conclusions are drawn and mentioned at the end of the section.")

    st.subheader('Listing Price and Categorical Features Analysis (per City by Day of Week or Room Type):')

    col1,col2 = st.columns(2, gap='large')
    with col1:
        df_city_day=df.groupby(by=['city','day_of_week'])['listing_price'].mean().reset_index().sort_values(by=['listing_price'])
        df_city_day['listing_price']=round(df_city_day['listing_price'],0)
        city_day_fig=px.bar(df_city_day, x='city', y='listing_price', color='day_of_week', barmode='group', title = 'Average Listing Prices per City by Day of Week', text='listing_price', labels={'city': 'City', 'day_of_week': 'Day of the week', 'listing_price': 'Price' })
        city_day_fig.update_traces(texttemplate='%{text:2s}', textposition='outside')
        st.plotly_chart(city_day_fig,use_container_width=True)

    with col2:
        
        df_city_room=df.groupby(by=['city','room_type'])['listing_price'].mean().reset_index().sort_values(by=['listing_price'])
        df_city_room['listing_price']=round(df_city_room['listing_price'],0)
        city_room_fig=px.bar(df_city_room, x='city', y='listing_price', color='room_type', barmode='group', title = 'Average Listing Prices per City by Room Type', text='listing_price', labels={'city': 'City', 'room_type': 'Room Type', 'listing_price': 'Price' })
        city_room_fig.update_traces(texttemplate='%{text:2s}', textposition='outside')
        st.plotly_chart(city_room_fig,use_container_width=True)

    st.subheader(city_selection +' Listing Price (Target) / Features Regression Analysis:')
    st.markdown('**Please select a a :blue[city] in the sidebar.**')
    df_city=df.loc[df['city']==city_selection]

    col1,col2=st.columns([1,4])
    columns_list=['day_of_week','host_is_superhost','multiple_rooms ',' business_facilities']
    xscale_list=['person_capacity','cleanliness_rating','guest_satisfaction_overall','bedrooms_quantity', 'city_center_distance', 'metro_distance']

    with col1:
        x_scale = st.selectbox("Select a feature for x scale:", tuple(xscale_list))
        z_scale = st.selectbox("Select a feature for columns:", tuple(columns_list))

    with col2:
        
        fig_regression=px.scatter(df_city, x=x_scale, y='listing_price', color = df_city['room_type'], hover_data=['guest_satisfaction_overall','listing_price','bedrooms_quantity','cleanliness_rating'],
           labels={'room_type' : 'Room Type','listing_price': 'Listing Price', 'day_of_week': 'Day of Week', 'bedrooms_quantity':'Number of Bedrooms'},
              color_continuous_scale=px.colors.sequential.Viridis,
              facet_col=z_scale, title='Target/Feature Regression Analysis')

        st.plotly_chart(fig_regression,use_container_width=True)    
   
    st.subheader(city_selection+  ' Listing Price (Target) / Features Correlation Analysis:')
    
    col1,col2 = st.columns(2, gap='large')
    with col1:
        corr_df = df_city.corr()[['listing_price']].sort_values(by='listing_price', ascending=False)

        st.markdown(' ##### Target/Features Correlations:')
        fig = px.imshow(corr_df.round(2), color_continuous_scale='Viridis', aspect="auto", text_auto=".2f")
        fig.update_xaxes(side="top")
        st.plotly_chart(fig,use_container_width=True)
        st.caption('**NOTE: The correlation between the Target and the Features as well as between the Features is :blue[quite low]. Therefore, Neuaral Network and Random Forest Machine Learning Models will be selected. Also, as the correlation between the features is low, including all the features in the machine learning model  will likely result in smaller errors**')

    with col2:
        st.markdown(' ##### Features Correlations:')

        corr_m = df_city.corr()

         # Correlation
        df_corr =corr_m.round(2)
        # Mask to matrix
        mask = np.zeros_like(df_corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        # Viz
        df_corr_viz = df_corr.mask(mask).dropna(how='all').dropna('columns', how='all')
        fig = px.imshow(df_corr_viz, text_auto=True, color_continuous_scale='Viridis')
        # fig.update_xaxes(side="top")
        st.plotly_chart(fig,use_container_width=True) 
     
with tab4:
    icon,title=st.columns([1,20])
    with icon: 
        st.image('Images/house.png', use_column_width=True)
    with title:
        st.title('Machine Learning Model Selection')
    
    with st.expander("About this section"):
        st.markdown("Machine Learning section allows the user to select and calibrate a machine learning model by adjusting the features of the dataset to be included in the machine learning and the machine learning parameters. The section proceded with training the selected model, predicting the results and evaluating the model performance. The predictions are then copared to the actual lisitng prices and the model is saved. Different models evaluation metices can be compared and a better performing model is selected for the price predictions based on those evaluations.")


    df_predict=df.copy()

    y = df['listing_price']

    features_selection=df.drop(columns=['listing_price'])

    st.header('Features Selection:')

    features_list=st.multiselect('Select Features to be included in ML:', features_selection.columns, default = list(features_selection.columns))

    features_df=df[features_list]
    target_col,features_col=st.columns([1,8])
    with target_col:
        st.markdown('Target:')
        st.write(y[-5:],use_container_width=True)
    with features_col:
        st.markdown('Features (bottom 5 rows):')
        st.dataframe(features_df.tail())


    # Create a list of categorical variables 
    categorical_variables = list(features_df.dtypes[features_df.dtypes == "object"].index)

    # Create a dataframewith categorical variables 
    df_categrical=features_df[categorical_variables]
    
    # Create a dDataframe with non-categorical variables:
    df_numerical=features_df.drop(columns = categorical_variables)

    # Create a OneHotEncoder instance
    enc =OneHotEncoder(categories='auto', sparse=False)
    # Encode the categorcal variables using OneHotEncoder
    encoded_data =  enc.fit_transform(features_df[categorical_variables])

    # Create a DataFrame with the encoded variables
    encoded_df = pd.DataFrame(
        encoded_data,
        columns = enc.get_feature_names(categorical_variables)
    )

    # Add the numerical variables from the original DataFrame to the one-hot encoding DataFrame
    encoded_df = pd.concat(
        [
            df_numerical,
            encoded_df
        ],
        axis=1
    )

    # Define the features as y and X:
    X = encoded_df
     # Review the features
    st.markdown('Features Encoded Dataframe (bottom 5 rows):')
    st.dataframe(X.tail())

    # Split the preprocessed data into a training and testing dataset
    # Assign the function a random_state equal to 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # Create a StandardScaler instance
    scaler =  StandardScaler()

    # Fit the scaler to the features training dataset
    X_scaler = scaler.fit(X_train)

    # Fit the scaler to the features training dataset
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    create_model=st.button('Create Machine Learning Model?')
    if create_model:
        with st.spinner('Model is being created. Please wait for the complition confirmation message.'):
            time.sleep(5)

        if mlm_selection =='Neural Network':

            # Define the the number of inputs (features) to the model
            number_input_features =  len(X_train.iloc[0])
            # st.write(number_input_features)

            # Define the number of neurons in the output layer
            number_output_neurons = 1

            #  n_epochs
            # n_layers
            # Define the number of hidden nodes for the first hidden layer
            hidden_nodes_layer1 =   (number_input_features + number_output_neurons) // 2 
            # st.write(hidden_nodes_layer1)

            # Define the number of hidden nodes for the second hidden layer
            hidden_nodes_layer2 = (hidden_nodes_layer1 + number_output_neurons) // 2

            #   Review the number hidden nodes in the second layer
            # st.write(hidden_nodes_layer2)

            # Define the number of hidden nodes for the third hidden layer
            hidden_nodes_layer3 = (hidden_nodes_layer2 + number_output_neurons) // 2

            #   Review the number hidden nodes in the second layer
            # st.write(hidden_nodes_layer3)
            # Create the Sequential model instance
            nn = Sequential()
            # Add the first hidden layer
            nn.add(Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu"))

            if n_layers==2:
                # Add the second hidden layer
                nn.add(Dense(units=hidden_nodes_layer2, activation="relu"))
            elif n_layers==3:
                # Add the second hidden layer
                nn.add(Dense(units=hidden_nodes_layer2, activation="relu"))
                # Add the third hidden layer
                nn.add(Dense(units=hidden_nodes_layer3, activation="relu"))

            # Output layer
            nn.add(Dense(units=number_output_neurons, activation="linear"))

                #Compile the model
            nn.compile(loss="mean_squared_error", optimizer="adam", metrics=["mse"])

            # Fit the model
            model = nn.fit(X_train_scaled, y_train, epochs=n_epochs, verbose=1) 
            # Evaluate the model loss and accuracy metrics using the evaluate method and the test data
            loss, mse =  nn.evaluate(X_test_scaled,y_test,verbose=2)
            # st.write(round(np.sqrt(mse),2))
            # st.write(round(loss,2))
            prediction=nn.predict(X_test_scaled)
            mean_square=round(np.sqrt(mean_squared_error(y_test, prediction)),2)
            mean_abs = round(mean_absolute_error(y_test, prediction),2)
            st.success('Model is created and saved. View the evaluation results below:')
            st.markdown(f'Mean Square Error is: **:blue[{mean_square}]**')
            st.write(f'Mean Absolute Error is: **:blue[{mean_abs}]**')

            # Visualizing predictions vs lisitng price:
            df_predict['prediction']=nn.predict(X_scaler.transform(X))
            st.markdown('Dataframe including predictions:')
            st.dataframe(df_predict.tail())
            fig_predictions=px.line(df_predict, y= ['listing_price','prediction'],title='<b>Listing Price vs Predicted Price</b>')
            fig_predictions.update_layout(showlegend=True, yaxis_title="Prices", xaxis_title="Data Points")
            st.plotly_chart(fig_predictions,use_container_width=True)


            # Save model as JSON
            nn_json = nn.to_json()

            file_path = Path("./Resources/model.json")
            with open(file_path, "w") as json_file:
                json_file.write(nn_json)
            # Save weights
            file_path = "./Resources/model.h5"
            nn.save_weights("./Resources/model.h5")
   
            encoded_df.to_csv(Path('./Resources/neural_network.csv'), encoding='utf-8', index=False)

        else:
            # Create arandom forest regressor model

            rf_model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, random_state=1)
            
            # Fit the model
            rf_model = rf_model.fit(X_train_scaled, y_train)

           # Making predictions using the testing data
            prediction = rf_model.predict(X_test_scaled)

            # Displaying results            
            mean_square=round(np.sqrt(mean_squared_error(y_test, prediction)),2)
            mean_abs = round(mean_absolute_error(y_test, prediction),2)
            st.success('Model is created and saved. View the evaluation results below:')
            st.markdown(f'Mean Square Error is: **:blue[{mean_square}]**')
            st.write(f'Mean Absolute Error is: **:blue[{mean_abs}]**')

            # Feature Importance
            # Random Forests in sklearn will automatically calculate feature importance
            importances = rf_model.feature_importances_
            # Zip the feature importances with the associated feature name
            important_features = zip(X.columns, importances)

            # Create a dataframe of the important features
            importances_df = pd.DataFrame(important_features)

            # Rename the columns
            importances_df = importances_df.rename(columns={0: 'Feature', 1: 'Importance'})

            # Set the index
            importances_df = importances_df.set_index('Feature')

            # Sort the dataframe by feature importance
            importances_df = importances_df.sort_values(by='Importance',ascending=False)

            # st.dataframe(importances_df)
            # Create a plot
            fig_importance = px.bar(importances_df,  title = 'Feature Importance')
            fig_importance.update_layout(uniformtext_minsize=8, yaxis_title='Importance Score', xaxis_title='Feature', showlegend=False)
            st.plotly_chart(fig_importance,use_container_width=True)

            # Visualizing predictions vs lisitng price:
            df_predict['prediction']=rf_model.predict(X_scaler.transform(X))
            st.markdown('Dataframe including predictions:')
            st.dataframe(df_predict.tail())
            fig_predictions=px.line(df_predict, y= ['listing_price','prediction'],title='<b>Listing Price vs Predicted Price</b>')
            fig_predictions.update_layout(showlegend=True, yaxis_title="Prices", xaxis_title="Data Points")
            st.plotly_chart(fig_predictions,use_container_width=True)

            # Save model
            # data = {'model': rf_model, 'encoder': enc, 'scaler': scaler_rf}
            data = {'model': rf_model, 'scaler': X_scaler}
            with open(Path('./Resources/random_forest.pkl'), 'wb') as file:
                pickle.dump(data, file)

            # save the features_df dataframe as csv for future reference:
            X.to_csv(Path('./Resources/rf.csv'), encoding='utf-8', index=False)
            # st.dataframe(X)

with tab5:

    icon,title=st.columns([1,20])
    with icon: 
        st.image('Images/price.png', use_column_width=True)
    with title:
        st.title('Listing Price Predictor') 

    with st.expander("About this section"):
        st.markdown("Listing Price Prediction takes the users input about threir airbnb property and applies the model created in the Machine Learning section to predict the listing price. User inputs are also demonstrated next to the predicted price.")

    st.subheader('Applying Saved Machine Learning Model:')
    st.markdown('**Random Forest Model showed to be superior to Neural Network in terms of producing smaller errors, as well as being less time consuming to run. Therefore, Random Forest will be used to generate Airbnb price prediction in this section.**')
    st.markdown('**Provide information about your Airbnb on the Sidebar to predict the lising price**')
    st.warning('**Please make sure the Random Forest model is saved in the Resources folder before predicting the price! If not saved, go to the Machine Learning tab and create the model.**')
    st.error('**Do not start predicting before all the required Airbnb relating input is provided!**')
    
    agree = st.button('Predict listing price?')

    if agree:
        # download the encoded complete features data to 1) align the selected features when generating the model and user inout and 2) train the scaler
        file_path = Path("./Resources/rf.csv")
    
        predict_df = pd.read_csv(file_path)
        model_features = predict_df.columns.to_list()
        # st.write(model_features)
        #adjust the columns of user input df to the features selected when creating the model:
        input_df=input_df[model_features]
        # st.dataframe(predict_df.tail())
        # st.dataframe(input_df)
        X_rf=predict_df

        # Create a StandardScaler instance
        scaler_rf =  StandardScaler().fit(X_rf)

        # Fit the scaler to the features training dataset


        # Fit the scaler to the features training dataset
        X_rf_scaled = scaler_rf.transform(X_rf)
        X_user = scaler_rf.transform(input_df)
        # st.write(X_rf_scaled)
        # st.write(X_user)

        # load Random Forest model
        with open(Path('./Resources/random_forest.pkl'), 'rb') as file:
                data=pickle.load(file)

        loaded_model=data['model']

        
        # st.write(X_scaler.transform(X_final))
        one_row = input_df.iloc[0].values
        X_one_row = scaler_rf.transform([one_row])
        lisitng_price=loaded_model.predict(X_one_row)[0]
        input_df["predicted"] = loaded_model.predict(X_user)
                    
        col1,col2=st.columns(2, gap='large')
        with col1:
            st.markdown(f'#### **Your Airbnb lisitjng price is predicted to equal EUR: :blue{lisitng_price: .2f}**')
        with col2:
            st.markdown('Airbnb Characteristics (including predicted price):')
            st.dataframe(input_df.T, use_container_width=True)
        
                
    