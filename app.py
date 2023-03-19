import pickle
import time
from pathlib import Path

import chart_studio.plotly as py
import cufflinks as cf
import hvplot.pandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
import tensorflow as tf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, model_from_json

# init_notebook_mode(connected=True)
# cf.go_offline()

#create internet page name
# st.set_page_config(page_title="AirBnB Price Prediction with ML", layout='wide', initial_sidebar_state="collapsed")

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
    # with st.spinner('Application running, please wait'):
    #     time.sleep(25)
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


    st.header('Original Data Review and Cleansing:')
    # df_original = get_data()

    st.subheader('Original Dataset (top 5 rows) :')
    st.dataframe(df_original.head(),use_container_width=True)

    #make a copy of the original dataframe
    # bnb_df=df_original.copy()

    # remove redundant columns
    bnb_df=bnb_df.drop(columns=['room_shared','room_private','attr_index','attr_index_norm','rest_index','rest_index_norm', 'lng', 'lat'])

    # bnb_df=bnb_df.drop(columns=['room_shared','room_private','attr_index','attr_index_norm','rest_index','rest_index_norm'])    

    # rename columns with more descriptive names:
    bnb_df.rename(columns = {'day':'day_of_week','realSum':'listing_price', 'multi':'multiple_rooms ',
                              'biz':' business_facilities', 'bedrooms':'bedrooms_quantity', 'dist':'city_center_distance', 'metro_dist':'metro_distance'}, inplace = True)

    #modify columns with Boolean data:
    bnb_df['host_is_superhost']=bnb_df['host_is_superhost'].apply(lambda x:1 if x==True else 0)

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
    # outliers=st.button('Show Outliers Analysis?')
    # if outliers:
     # select column to analize value_counts()
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
        # price_outliers=px.violin(bnb_df, x='city', y='listing_price', box=True, 
        #         color='day_of_week', points='all', hover_data=bnb_df.columns,
        #         labels={'city': 'City','listing_price': 'Listing Price' }, title = 'Listing Price Distribution per city')
        # st.plotly_chart(price_outliers,use_container_width=True)
        # st.markdown(f' ##### Total Listing Price Quantiles')
        # st.write(bnb_df['listing_price'].quantile([0.01, 0.25, 0.5, 0.75, 0.99]))
        fig_quantile=px.bar(bnb_df.loc[bnb_df['city']==city_selection]['listing_price'].quantile([0.01, 0.25, 0.5, 0.75, 0.99]),labels={'index': 'Quantile', 'value': 'Listing Price' }, title = city_selection + ' - Listing Price Quantile Distribution')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig_quantile,use_container_width=True)

        st.caption('Based on the descriptive statistics of the Dataframe and on the distribution analysis the lisiting prices above **:blue[650]** can be considered as outliers and will be removed for the Exploratory Data Analysis.')

    with col2:
        # city_selection = st.selectbox("Select a city to analyse further the listing price outliers:", tuple(bnb_df['city'].unique()))

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
        # st.write(bnb_df.loc[bnb_df['city']==city_selection]['listing_price'].quantile([0.01, 0.25, 0.5, 0.75, 0.99]))
    
    # remove outliers (listing prices above 650):
    df=bnb_df.loc[bnb_df['listing_price']<=650]
    df = df.reset_index(drop=True)
    st.markdown('#### Dataset Shape corrected for outliers:')
    st.write(df.shape,use_container_width=True)
    # st.dataframe(df.tail(),use_container_width=True)
    price_outliers_new=px.violin(df, x='city', y='listing_price', box=True, 
                color='day_of_week', points='all', hover_data=df.columns,
                labels={'city': 'City','listing_price': 'Listing Price' }, title = 'Listing Price Distribution per city (corrected for outliers)')
    st.plotly_chart(price_outliers_new,use_container_width=True)
   
    
with tab3:
    # # Create a list of categorical variables 
    # categorical_variables = list(df.dtypes[df.dtypes == "object"].index)
    # # Create a dataframewith categorical variables 
    # df_categrical=df[categorical_variables]
    
    # # Create a dDataframe with non-categorical variables:
    # df_numerical=df.drop(columns = categorical_variables)

    icon,title=st.columns([1,20])
    with icon: 
        st.image('Images/rent.png', use_column_width=True)
    with title:
        st.title('Exploratory Data Analysis')


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
    st.caption('**Please select a a :blue[city] in the sidebar.**')
    df_city=df.loc[df['city']==city_selection]

    col1,col2=st.columns([1,4])
    columns_list=['day_of_week','host_is_superhost','multiple_rooms ',' business_facilities']
    xscale_list=['person_capacity','cleanliness_rating','guest_satisfaction_overall','bedrooms_quantity', 'city_center_distance', 'metro_distance']
    # df_regression=df.copy()

    with col1:
        x_scale = st.selectbox("Select a feature for x scale:", tuple(xscale_list))
        z_scale = st.selectbox("Select a feature for columns:", tuple(columns_list))
        # color = st.selectbox("Select a feature for color:", tuple(df['city'].columns))
    with col2:
        
        fig_regression=px.scatter(df_city, x=x_scale, y='listing_price', color = df_city['room_type'], hover_data=['guest_satisfaction_overall','listing_price','bedrooms_quantity','cleanliness_rating'],
           labels={'room_type' : 'Room Type','listing_price': 'Listing Price', 'day_of_week': 'Day of Week', 'bedrooms_quantity':'Number of Bedrooms'},
              color_continuous_scale=px.colors.sequential.Viridis,
              facet_col=z_scale, title='Target/Feature Regression Analysis')

        st.plotly_chart(fig_regression,use_container_width=True)    

       # fig_regression_features=px.scatter(df_city, x=x_scale, y=y_scale,  hover_data=['guest_satisfaction_overall','listing_price','bedrooms_quantity','cleanliness_rating'],
        #    labels={'room_type' : 'Room Type','listing_price': 'Listing Price', 'day_of_week': 'Day of Week', 'bedrooms_quantity':'Number of Bedrooms'}, title='Features Regression Analysis')

    
    st.subheader(city_selection+  ' Listing Price (Target) / Features Correlation Analysis:')
    
    col1,col2 = st.columns(2, gap='large')
    with col1:
        corr_df = df_city.corr()[['listing_price']].sort_values(by='listing_price', ascending=False)
        # fig_corr=plt.figure()
        # sns.heatmap(corr_df.round(2), annot=True, vmin=-1, vmax=1, center =0, cmap='vlag')
        # sns.set_theme(style='dark')
        # st.pyplot(fig_corr,use_container_width=True)  
        st.markdown(' ##### Target/Features Correlations:')
        fig = px.imshow(corr_df.round(2), color_continuous_scale='Viridis', aspect="auto", text_auto=".2f")
        fig.update_xaxes(side="top")
        st.plotly_chart(fig,use_container_width=True)
        st.caption('**NOTE: The correlation between the Target and the Features as well as between the Features is :blue[quite low]. Therefore, Neuaral Network and Random Forest Machine Learning Models will be selected.**')

    with col2:
        st.markdown(' ##### Features Correlations:')

        corr_m = df_city.corr()
        # fig_corr=plt.figure()
        # sns.heatmap(corr_m.round(2), annot=True, vmin=-1, vmax=1, center =0, cmap='vlag')
        # sns.set_theme(style='dark')
        # st.pyplot(fig_corr,use_container_width=True)  

        # fig = px.imshow(corr_m.round(2), color_continuous_scale='Viridis', aspect="auto", text_auto=".2f")
        # fig.update_xaxes(side="top")

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
    # with col1:
    #     x_scale = st.selectbox("Select a feature for x scale:", tuple(bnb_df.columns))
    #     y_scale = st.selectbox("Select a feature for y scale:", tuple(bnb_df.columns))
    #     z_scale = st.selectbox("Select a feature for z scale:", tuple(bnb_df.columns))
    #     color = st.selectbox("Select a feature for color:", tuple(bnb_df.columns))
    # with col2:
    #     fig_3d = px.scatter_3d(df, x=x_scale, y=y_scale,  z=z_scale, color = color, opacity=0.7, hover_data=['guest_satisfaction_overall','listing_price','bedrooms_quantity','cleanliness_rating'],
    #         labels={'room_type' : 'Room Type', 'listing_price': 'Listing Price' })
    #     st.plotly_chart(fig_3d,use_container_width=True)
    # city_selection2 = st.selectbox("Select a city:", tuple(df['city'].unique()))
    # st.markdown(' ####  Day of Week:')
    # weekday_selection = st.selectbox("Select a day of week :", tuple(df['day_of_week'].unique()))
    # st.markdown(' #### Select Room Type:')
    # room_selection = st.selectbox("Select a room type :", tuple(df['room_type'].unique()))

    # st.header('Original Data Review and Cleansing:')





    
with tab4:
    icon,title=st.columns([1,20])
    with icon: 
        st.image('Images/house.png', use_column_width=True)
    with title:
        st.title('Machine Learning Model Selection')
    
    
    # st.markdown('#### Dataset Shape:')
    # st.write(df.tail(),use_container_width=True)
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

    # create an empty dataframe to store the names of the selected columns
    empty_df = pd.DataFrame(columns=features_df.columns)

    # Create a list of categorical variables 
    categorical_variables = list(features_df.dtypes[df.dtypes == "object"].index)

    # Create a dataframewith categorical variables 
    df_categrical=features_df[categorical_variables]
    
    # Create a dDataframe with non-categorical variables:
    df_numerical=features_df.drop(columns = categorical_variables)

    # st.dataframe(df_categrical.tail())
    # st.dataframe(df_numerical.tail())

#     le = LabelEncoder()
#     features[categorical_variables] = features[categorical_variables].apply(lambda col: le.fit_transform(col))
# #   Viewing first few rows of data
#     st.markdown('test')
#     st.dataframe(features[categorical_variables].tail(100))

#     # st.write(features['city'].value_counts())
#     # st.write(df['city'].value_counts())
#     onehotencoder = OneHotEncoder(sparse=False)
#     encoded_data_1=onehotencoder.fit_transform(features[categorical_variables])
#     encoded_df_1 = pd.DataFrame(
#         encoded_data_1,
#         columns = onehotencoder.get_feature_names(categorical_variables)
#     )
#     st.markdown('test2')
#     st.dataframe(encoded_df_1.tail(10))

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

    # # Fit the scaler to the target training dataset
    # y_scaler = scaler.fit(y_train)

    # # Fit the scaler to the target training dataset
    # y_train_scaled = y_scaler.transform(y_train)
    # y_test_scaled = y_scaler.transform(y_test)

    # st.write(y_test_scaled[-5:])
    # st.write(X_test_scaled[-5:])
    create_model=st.button('Create Machine Learning Model?')
    if create_model:
        with st.spinner('Model is being created. Please wait for the complition confirmation message.'):
            time.sleep(5)

        if mlm_selection =='Neural Network':
            # Create a deep neural network by assigning the number of input features, the number of layers, and the number of neurons on each layer using Tensorflow’s Keras.

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

            # Save model as JSON
            nn_json = nn.to_json()

            file_path = Path("./Resources/model.json")
            with open(file_path, "w") as json_file:
                json_file.write(nn_json)
            # Save weights
            file_path = "./Resources/model.h5"
            nn.save_weights("./Resources/model.h5")

            # save the emptu dataframe as csv for future reference:
            empty_df.to_csv(Path('./Resources/neural_network.csv'), encoding='utf-8', index=False)

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

            # Save model
            data = {'model': rf_model, 'encoder': enc, 'scaler': scaler}
            with open(Path('./Resources/random_forest.pkl'), 'wb') as file:
                pickle.dump(data, file)

            # save the emptu dataframe as csv for future reference:
            empty_df.to_csv(Path('./Resources/random_forest.csv'), encoding='utf-8', index=False)

with tab5:

    icon,title=st.columns([1,20])
    with icon: 
        st.image('Images/price.png', use_column_width=True)
    with title:
        st.title('Listing Price Predictor') 

    st.subheader('Applying Saved Machine Learning Model to User Input to Predict a Listing Price:')
    # response=st.radio('Chose a model to predict a listing price:', tuple(['Random Forest','Neural Network']))
    st.markdown('**Select a model and provide input about your Airbnb on the Sidebar**')
    if mlm_selection == 'Random Forest':
        file_path = Path("./Resources/random_forest.csv")
        # lod model
        with open(Path('./Resources/random_forest.pkl'), 'rb') as file:
                data=pickle.load(file)

        loaded_model=data[ 'model']
        enc=data[ 'encoder']
        scaler=data[ 'scaler']
    else: 
        file_path = Path("./Resources/neural_network.csv")
        # load model:
        file_path = Path("./Resources/model.json")
        with open(file_path, "r") as json_file:
            model_json = json_file.read()
        loaded_model = model_from_json(model_json)

        # load weights into new model
        file_path = "./Resources/model.h5"
        loaded_model.load_weights(file_path)
    
    empty_df = get_data(file_path)
    feater_columns = empty_df.columns.to_list()

    st.subheader('Airbnb Characteristics (User Input Summary):')
        
    input_dict={}
    st.sidebar.subheader('Airbnb Charcteristics:')
    for column in feater_columns:
        if column == "guest_satisfaction_overall":
            input_dict[column] = st.sidebar.slider(column, min_value=1, max_value=100, value=90, step = 1)
        elif column == "metro_distance":
            input_dict[column] = st.sidebar.slider(column, min_value=0.01, max_value=14.99, value=1.01, step = 0.01)
        elif column =="city_center_distance":
            input_dict[column] = st.sidebar.slider(column, min_value=0.01, max_value=29.99, value=1.01, step = 0.01)
        elif column in categorical_variables:
            input_dict[column] =st.sidebar.selectbox(column, tuple(df[column].sort_values().unique()))          
        else:
            input_dict[column] =float(st.sidebar.selectbox(column, tuple(df[column].sort_values().unique())))

    input_df=pd.DataFrame(data = input_dict, index=[0])
     # st.write(input_df)
    st.write(input_dict)

    # Create a list of categorical variables 
    categorical_variables = list(input_df.dtypes[df.dtypes == "object"].index)

    # # Create a OneHotEncoder instance
    # enc =OneHotEncoder(categories='auto', sparse=False)

    # Encode the categorcal variables using OneHotEncoder
    encoded_data =  enc.fit_transform(input_df[categorical_variables])

    # Create a DataFrame with the encoded variables
    encoded_df = pd.DataFrame(
        encoded_data,
        columns = enc.get_feature_names(categorical_variables)
    )

    # Add the numerical variables from the original DataFrame to the one-hot encoding DataFrame
    encoded_df = pd.concat(
        [
            input_df.drop(columns = categorical_variables),
            encoded_df
        ],
        axis=1
    )

    # Define the features as y and X:
    X = encoded_df
     # Review the features
    st.dataframe(X)
    X_scaled= scaler.transform(X)
    st.dataframe(X_scaled)
    # lisitng_price=loaded_model.predict(X_scaled)[0][0]

    st.subheader('Listing Price Prediction:')
    st.markdown(f'#### **Your Airbnb lisitjng price is predicted to be EUR: :blue[{50000}]**')
    
 