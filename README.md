# Movie Recommendation System - ML Coding Challenge - Streamlit

## Overview

This Movie Recommendation System is built using Python and deployed on the cloud with Streamlit. The system utilizes the Surprise library for collaborative filtering and integrates data exploration with Pandas, Matplotlib, and Seaborn.

## Key Features

1. **Data Overview:** Displays an overview of the movies and ratings datasets, merged data, and a summary of ratings distribution.

2. **Exploratory Data Analysis (EDA):** Provides insights into the top-rated movies, genres distribution, and visualizations of ratings.

3. **Feature Enhancement:** Introduces additional features such as average rating per movie, number of ratings per movie, and release year.

4. **Test Train Split Overview:** Demonstrates the process of splitting data into training and testing sets for model validation.

5. **Recommendation Abstract:** Trains a collaborative filtering recommendation system using Surprise's SVD algorithm. Evaluates the model performance and offers movie recommendations based on user input genres.

## Cloud deployed URL (streamlit)

- https://ml-coding-challenge-recommendation-system.streamlit.app/

## How to Run on local

1. **Dependencies:** Ensure you have the necessary dependencies by running `pip install -r requirements.txt`.

2. **Run the App:** Execute `streamlit run app.py` in the terminal to launch the Streamlit app.

3. **Explore Recommendations:** Navigate through the app sections using the sidebar to explore data, perform EDA, enhance features, and experience movie recommendations.


## Acknowledgments

- This project was created as part of a technical assessment.
- Inspired by real-world recommendation systems.
- Special thanks to Kaggle for providing the movie recommendation dataset.
