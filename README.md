# Airbnb Listing Price Predictor

### One of the critical components determining business competitiveness is a sound pricing strategy. New businesses need to set a right price to be able to enter the market, while both new and existing companies use pricing as one of their marketing mix tools to stay competitive and optimize their revenues.<br/> Airbnb Price Predictor, powered by Machine Learning Models and various data analysis techniques, provides Airbnb owners with a a price predicting tool based on their properties characteristics. The tool also allows the property owners to understand the interdependencies between the different components of their offerings.<br/> This application can also be used by the travelers, who would like to estimate the Airbnb prices in their trip destination.

---

![airbnb](Images/airbnb.jpg)

---

## Table of contents

1. [Technologies](#technologies)
2. [Installation Guide](#installation-guide)
3. [Usage](#usage)
4. [Contributors](#contributors)
5. [License](#license)

---

## Technologies

`Python 3.9`

_Libraries_

1. `Pandas` is a Python package that provides fast, flexible, and expressive data structures designed to make working with large sets of data easy and intuitive.

   - [pandas](https://github.com/pandas-dev/pandas) - for the documentation, installation guide and dependencies.

2. `PyViz` is a Python visualization package that provides a single platform for accessing multiple visualization libraries. The visualization libraries used in this application are listed below: <br/>

   - [PyViz ](https://pyviz.org/) - for guidance on how to start visualization, interactive visualization, styles and layouts customization.

   - [Plotly](https://plotly.com/) - is a library that allows developers to build interactive charts, tables and graphs from data.

3. `Streamlit` is a library that allows developers to build web applications with live user input.

   - [Streamlit](https://streamlit.io/) - to read more about deploying, installing and customizing.<br/>

4. `NumPy` is an open source library that adds computational tools to a project, including mathematical functions and random number generators.

   - [Numpy](https://numpy.org/) - to read about available functions and installation.<br/>

5. `Scikit-learn` is a simple and efficient tools for predictive data analysis. It is built on NumPy, SciPy, and matplotlib.

   - [scikit-learn ](https://scikit-learn.org/stable/) - for information on the library, its features and installation instructions.<br/>

---

## Installation Guide

Start the app here: [Airbnb Listing Price Predictor](https://borisdudkin-fintech-project-2-airbnb-price-predictio-app-bz4phm.streamlit.app/)<br/>

Alternatively, the application can be started from the terminal using Streamlit, once in the directory of the application and all the required libraries and the application are installed locally (see instructions below):<br/>

```python
streamlit run app.py
```

### Library Installations

Before using the application first install the following dependencies by using your terminal:

> Option 1 - Intsall needed libraries with requirements.txt file included in the reporsitory. For this option:<br/>

- install [pipreqs](https://pypi.org/project/pipreqs/):

  ```python
  pip install pipreqs
  ```

- Install all libraries that we have in requirements.txt::

  ```python
  pip install -r requirements.txt:
  ```

> Option 2 - Install each library individually:<br/>

To install pandas run:

```python
pip install pandas
```

```python
# or conda
conda install pandas
```

To install the PyViz visualizations, in Terminal run:

```python
# conda
conda install -c pyviz bokeh holoviews plotly
```

Confirm the installation of all the PyViz packages by running the following commands in Terminal:

```python
conda list pyviz
```

```python
 conda list plotly
```

To install Streamlit, in Terminal run:

```python
pip install streamlit
```

Confirm the installation of the Streamlit package by running the following commands in Terminal:

```python
 conda list streamlit
```

To install numpy, in Terminal run:

```python
pip install numpy
```

Confirm the installation of the numpy package by running the following commands in Terminal:

```python
 conda list numpy
```

To install scikit library, in Terminal run:

```python
# PuPi
pip install -U scikit-learn
```

To install compressor for pickle in Terminal run:

```python
pip install mgzip
```

---

## Usage

> Application summary<br/>

The Airbnb Price Predictor provides tools needed to analyse Airbnb related data and arrive at a property price with the help of Machine Learning Models.<br/>
Furthermore, Streamlit library transforms Airbnb Price Predictor into an interactive web application that nontechnical users without any coding experience can use.<br/>Finally, the tool takes the users through the whole experience of applying the Machine Learning techniques to a dataset.<br/>The following parts are covered:

- Introduction:<br/>

  - Introduction section provide information about the project, project's objective, dataset and also suggests other areas for future research.<br/>

    ![](Images/Introduction.gif)

- Original Data Review and Cleansing:<br/>

  - Initial Data Analysis begins with demonstrating the original dataset. It then continues by showing the progress in data cleansing - removing redundant features, giving columns more descriptive names, analizing the descriptive statistics of the data, its shape, checking for missing values duplicates and the types of data prsent in the dataset. The section continues with univariate analysis of the features and focuses on the outliers for our target variable _lisitng price_. The correction for outliers is suggested and the dataset after the correction is demostrated breaking down the information by city.<br/>
    ![](Images/Initial_data_analysis.gif)

- Exploratory Data Analysis:<br/>

  - Exploratory Data Analysis continues researching the data but this time focusing on the bivariate and multivariate relationship between the features and the target, as well as between different features, including categorical/numerical and numerical/numerical analysis. The conclusions are drawn and mentioned at the end of the section.<br/>
    ![](Images/Exploratory_data.gif)

- Machine Learning Model Selection:<br/>

  - Machine Learning section allows the user to select and calibrate a machine learning model by adjusting the features of the dataset to be included in the machine learning and the machine learning parameters. The section proceded with training the selected model, predicting the results and evaluating the model performance. The predictions are then copared to the actual lisitng prices and the model is saved. Different models evaluation metices can be compared and a better performing model is selected for the price predictions based on those evaluations.<br/>
    ![](Images/ML_1.gif)
    Evaluation and Prediction:
    ![](Images/ML_2.gif)

- Listing Price Predictor:<br/>

  - Listing Price Prediction takes the users input about threir airbnb property and applies the model created in the Machine Learning section to predict the listing price. User inputs are also demonstrated next to the predicted price.<br/>
    ![](Images/price_predict.gif)

> Getting started<br/>

- To use the Airbnb Price Predictor first clone the repository to your PC.<br/>
- Use `streamlit run app.py` as per the instructions in the [Installation Guide](#installation-guide) to run the application.<br/>

---

## Contributors

Contact Details:

Boris Dudkin:

- [Email](boris.dudkin@gmail.com)
- [LinkedIn](www.linkedin.com/in/Boris-Dudkin)

---

## License

MIT

---
