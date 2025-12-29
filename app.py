import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the movie data"""
    df = pd.read_csv("movies.csv", lineterminator="\n")
    
    # Preprocessing
    df["Release_Date"] = pd.to_datetime(df["Release_Date"]).dt.year
    df = df.drop(["Poster_Url", "Vote_Count"], axis=1)
    
    # Categorize movies
    stats = df['Vote_Average'].describe()
    q1, q2, q3 = stats['25%'], stats['50%'], stats['75%']
    
    def categorize_vote(vote):
        if vote >= q3:
            return 'Popular'
        elif vote >= q2:
            return 'Average'
        elif vote >= q1:
            return 'Below Average'
        else:
            return 'Not Popular'
    
    df['Category'] = df['Vote_Average'].apply(categorize_vote)
    df = df.drop(["Vote_Average"], axis=1)
    
    # Create combined features for recommendation
    df['combined_features'] = df['Overview'] + " " + df['Genre']
    df['title_clean'] = df['Title'].astype(str).str.lower().str.strip()
    
    return df

@st.cache_data
def build_similarity_matrix(df):
    """Build TF-IDF and cosine similarity matrix"""
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def recommend_movies(df, cosine_sim, movie_name, num_recommendations=5):
    """Get movie recommendations based on similarity"""
    movie_name = movie_name.lower().strip()
    
    matches = df[df['title_clean'].str.contains(movie_name, na=False)]
    
    if matches.empty:
        return None, "Movie not found. Try another title."
    
    idx = matches.index[0]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    top_movies = similarity_scores[1:num_recommendations+1]
    recommended_indices = [i[0] for i in top_movies]
    
    return df.iloc[recommended_indices], None

# Load data
df = load_and_preprocess_data()
cosine_sim = build_similarity_matrix(df)

# Main UI
st.title("🎬 Movie Recommendation System")
st.markdown("---")

# Sidebar
st.sidebar.header("Search Movies")
movie_search = st.sidebar.text_input("Enter a movie name:", placeholder="e.g., Avengers")
num_recommendations = st.sidebar.slider("Number of recommendations:", min_value=3, max_value=10, value=5)

# Main content
if movie_search:
    recommended_movies, error = recommend_movies(df, cosine_sim, movie_search, num_recommendations)
    
    if error:
        st.error(error)
    else:
        st.success(f"Found '{df[df['title_clean'].str.contains(movie_search.lower().strip(), na=False)].iloc[0]['Title']}' - Here are your recommendations:")
        st.markdown("---")
        
        # Display recommendations
        cols = st.columns(2)
        for idx, (_, movie) in enumerate(recommended_movies.iterrows()):
            with cols[idx % 2]:
                with st.container():
                    st.subheader(f"{idx + 1}. {movie['Title']}")
                    st.write(f"**Year:** {int(movie['Release_Date'])}")
                    st.write(f"**Genre:** {movie['Genre']}")
                    st.write(f"**Category:** {movie['Category']}")
                    st.write(f"**Popularity:** {movie['Popularity']:.2f}")
                    st.write(f"**Overview:** {movie['Overview'][:200]}...")
                    st.markdown("---")
else:
    st.info("👈 Enter a movie name in the sidebar to get recommendations")
    
    # Show some basic statistics
    st.markdown("---")
    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Movies", len(df))
    with col2:
        st.metric("Total Genres", df['Genre'].nunique())
    with col3:
        st.metric("Year Range", f"{int(df['Release_Date'].min())} - {int(df['Release_Date'].max())}")
    with col4:
        popular_count = len(df[df['Category'] == 'Popular'])
        st.metric("Popular Movies", popular_count)
    
    # Category distribution
    st.markdown("### Category Distribution")
    category_counts = df['Category'].value_counts()
    category_order = ['Popular', 'Average', 'Below Average', 'Not Popular']
    category_counts = category_counts.reindex(category_order, fill_value=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis', ax=ax)
    ax.set_title('Number of Movies per Category', fontsize=14)
    ax.set_xlabel('Category')
    ax.set_ylabel('Count')
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    st.pyplot(fig)


