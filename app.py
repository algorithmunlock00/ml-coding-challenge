import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import accuracy

movies_df = pd.read_csv('dataset/movies.csv')
rating_df = pd.read_csv('dataset/ratings.csv')

merged_df = pd.merge(rating_df, movies_df, on='movieId')


# Feature Enhancement: Average rating per movie
average_rating_per_movie = rating_df.groupby('movieId')['rating'].mean().reset_index()
average_rating_per_movie.columns = ['movieId', 'avg_rating']

# Feature Enhancement: Number of ratings per movie
num_ratings_per_movie = rating_df.groupby('movieId')['rating'].count().reset_index()
num_ratings_per_movie.columns = ['movieId', 'num_ratings']

# Feature Enhancement: Release Year
movies_df['release_year'] = movies_df['title'].str.extract(r'\((\d{4})\)$')

# Merge all features into one dataframe
movies_df_enhanced = pd.merge(movies_df, average_rating_per_movie, on='movieId')
movies_df_enhanced = pd.merge(movies_df_enhanced, num_ratings_per_movie, on='movieId')

combined_df = pd.merge(movies_df_enhanced, rating_df, on='movieId')

train_data, test_data = pd.DataFrame(columns=combined_df.columns), pd.DataFrame(columns=combined_df.columns)
unique_users = combined_df['userId'].unique()

for user_id in unique_users:
    user_data = combined_df[combined_df['userId'] == user_id]
    train_user, test_user = train_test_split(user_data, test_size=0.3, random_state=42)

    train_data = pd.concat([train_data, train_user])
    test_data = pd.concat([test_data, test_user])


# Page 1: Data Overview
def data_overview():
    st.title("Data Overview")
    
    st.write("Movies DataFrame:")
    st.dataframe(movies_df.head())
    
    st.write("Ratings DataFrame:")
    st.dataframe(rating_df.head())
    
    st.write("Merging both datasets based on common column 'movieId':")
    st.dataframe(merged_df.head())
    
    st.write("Summary of Ratings:")
    st.write(merged_df['rating'].describe())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='rating', data=merged_df, palette='viridis', ax=ax)
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    st.pyplot(fig)

def eda():
    st.title("Exploratory Data Analysis")

    # Ratings per movie
    ratings_per_movie = merged_df.groupby('title')['rating'].count().sort_values(ascending=False)
    st.write("Top 20 Movies with Highest Number of Ratings:")
    st.bar_chart(ratings_per_movie[:20])

    # Average rating per movie
    average_rating_per_movie = merged_df.groupby('title')['rating'].mean().sort_values(ascending=False)
    st.write("Top 20 Movies with Highest Average Ratings:")
    st.bar_chart(average_rating_per_movie[:20])

    # Distribution of movie genres
    genres_list = '|'.join(movies_df['genres']).split('|')
    genres_count = pd.Series(genres_list).value_counts()
    st.write("Distribution of Movie Genres:")
    st.bar_chart(genres_count)

    st.write("Merged DataFrame:")
    st.dataframe(merged_df.head())

    # Display the original code provided
    st.write("Original Code:")
    
    # Ratings per movie
    st.write("Ratings per movie:")
    st.write(ratings_per_movie)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ratings_per_movie[:20].plot(kind='bar', color='skyblue', ax=ax)
    plt.title('Top 20 Movies with Highest Number of Ratings')
    plt.xlabel('Movie Title')
    plt.ylabel('Number of Ratings')
    st.pyplot(fig)

    # Average rating per movie
    st.write("Average rating per movie:")
    st.write(average_rating_per_movie)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    average_rating_per_movie[:20].plot(kind='bar', color='salmon', ax=ax)
    plt.title('Top 20 Movies with Highest Average Ratings')
    plt.xlabel('Movie Title')
    plt.ylabel('Average Rating')
    st.pyplot(fig)

    # Distribution of movie genres
    st.write("Distribution of Movie Genres:")
    st.write(genres_count)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    genres_count.plot(kind='bar', color='lightgreen', ax=ax)
    plt.title('Distribution of Movie Genres')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    st.pyplot(fig)


# Page 2: Feature Enhancement
def feature_enhancement():
    st.title("Feature Enhancement")

    st.write("Added features:")
    st.write("- Average Rating per Movie:")
    st.dataframe(movies_df_enhanced[['movieId', 'avg_rating']].head())
    st.write("- Number of Ratings per Movie:")
    st.dataframe(movies_df_enhanced[['movieId', 'num_ratings']].head())
    st.write("- Release Year:")
    st.dataframe(movies_df_enhanced[['movieId', 'release_year']].head())

    st.write("Combined DataFrame with Rating Features:")
    st.dataframe(combined_df)

# Page 3: Test Train Split Overview
def test_train_split_overview():

    st.title("Test Train Split Overview")
    

    st.code(
        """
        train_df, test_df = train_test_split(
            combined_df,
            test_size=0.3,
            random_state=42
        )
        """,
        language="python"
    )

    st.write("Train Dataset:")
    st.dataframe(train_data)

    st.write("Test Dataset:")
    st.dataframe(test_data)

# Page 4: Recommendation Abstract
def recommendation_abstract():
    st.title("Recommendation Abstract")
    st.write("Training Recommendation System...")
    
    # Create Surprise reader
    reader = Reader(rating_scale=(0.5, 5))

    # Load training data for Surprise
    train_data_surprise = Dataset.load_from_df(train_data[['userId', 'movieId', 'rating']], reader).build_full_trainset()

    # Train the recommendation system using SVD algorithm
    algo = SVD()
    algo.fit(train_data_surprise)
    st.write("Recommendation System Trained Successfully!")

    st.title("Validation Performance")

    # Load test data for Surprise
    test_data_surprise = Dataset.load_from_df(test_data[['userId', 'movieId', 'rating']], reader).build_full_trainset().build_testset()

    # Generate predictions on the test data
    predictions = algo.test(test_data_surprise)

    # Evaluate the model performance using RMSE
    rmse = accuracy.rmse(predictions)
    st.write(f"Root Mean Squared Error (RMSE) on Test Data: {rmse:.4f}")

    st.title("Movie Recommendation")

    genre_list = [
    "Drama", "Comedy", "Thriller", "Action", "Romance", "Adventure", "Crime",
    "Sci-Fi", "Horror", "Fantasy", "Children", "Animation", "Mystery",
    "Documentary", "War", "Musical", "Western", "IMAX", "Film-Noir"
    ]

    st.write("Available Genres:")
    st.write(genre_list)

    # Get user input for favorite genres
    user_genres = st.text_input("Enter your favorite genres (comma-separated):")
    
    # Handle the case where no genres are provided
    if not user_genres:
        st.warning("Please enter at least one favorite genre.")
        return

    user_genres_list = user_genres.split(',')

    # Filter movies based on user genres
    user_movies = movies_df_enhanced[movies_df_enhanced['genres'].apply(lambda x: any(genre in x for genre in user_genres_list))]

    # Display user's favorite genres and top 5 recommended movies

    st.write("Your Favorite Genres:", user_genres_list)
    st.write("Top 5 Recommended Movies:")

    # Make predictions only if there are movies matching the user's genres
    if not user_movies.empty:
        recommendations = []

        for genre in user_genres_list:
            genre_movies = user_movies[user_movies['genres'].str.contains(genre)]
            
            for movie_id in genre_movies['movieId'].values:
                prediction = algo.predict(1, movie_id)  # No need to provide 'user' argument
                recommendations.append({
                    'title': genre_movies[genre_movies['movieId'] == movie_id]['title'].values[0],
                    'estimated_rating': prediction.est
                })

        # Sort recommendations by estimated rating in descending order
        recommendations = sorted(recommendations, key=lambda x: x['estimated_rating'], reverse=True)[:5]

        # Display the top 5 recommended movies
        for idx, recommendation in enumerate(recommendations, start=1):
            st.write(f"{idx}. {recommendation['title']} - Estimated Rating: {recommendation['estimated_rating']:.2f}")
    else:
        st.warning("No movies found for the provided genres.")

# Sidebar navigation
page_options = {
    "Data Overview": data_overview,
    "Exploratory Data Analysis": eda,
    "Feature Enhancement": feature_enhancement,
    "Test Train Split Overview": test_train_split_overview,
    "Recommendation Abstract": recommendation_abstract
}

# Display pages in the sidebar
selected_page = st.sidebar.radio("Movie Recommendation System", list(page_options.keys()), index=0)

# Display the selected page
page_options[selected_page]()
