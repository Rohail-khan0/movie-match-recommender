import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")

# Custom CSS for beautiful UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .movie-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

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

# Main UI Header
st.markdown('<h1 class="main-header">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Search Section
st.markdown("## 🔍 Search for Movie Recommendations")
col_search, col_slider = st.columns([3, 1])
with col_search:
    movie_search = st.text_input("Enter a movie name:", placeholder="e.g., Avengers, The Batman, Spider-Man", key="movie_search")
with col_slider:
    num_recommendations = st.slider("Number of recommendations:", min_value=3, max_value=10, value=5, key="num_recs")

st.markdown("---")

# Recommendations Section
if movie_search:
    recommended_movies, error = recommend_movies(df, cosine_sim, movie_search, num_recommendations)
    
    if error:
        st.error(f"❌ {error}")
    else:
        matched_movie = df[df['title_clean'].str.contains(movie_search.lower().strip(), na=False)].iloc[0]
        st.success(f"✅ Found **'{matched_movie['Title']}'** ({int(matched_movie['Release_Date'])}) - Here are similar movies you might enjoy:")
        st.markdown("---")
        
        # Display recommendations in a beautiful grid
        cols = st.columns(2)
        for idx, (_, movie) in enumerate(recommended_movies.iterrows()):
            with cols[idx % 2]:
                with st.container():
                    st.markdown(f"""
                    <div class="movie-card">
                        <h3 style="color: #667eea; margin-bottom: 0.5rem;">{idx + 1}. {movie['Title']}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.write(f"**📅 Year:** {int(movie['Release_Date'])}")
                        st.write(f"**⭐ Category:** {movie['Category']}")
                    with col_info2:
                        st.write(f"**📊 Popularity:** {movie['Popularity']:.2f}")
                    
                    # Genre badges
                    genres = movie['Genre'].split(', ')
                    genre_text = ' • '.join([f"`{g}`" for g in genres])
                    st.markdown(f"**🎭 Genre:** {genre_text}")
                    
                    # Overview
                    with st.expander("📖 Read Overview"):
                        st.write(movie['Overview'])
                    
                    st.markdown("---")
else:
    st.info("💡 **Enter a movie name above** to get personalized recommendations!")
    
    # Show some popular movies as suggestions
    st.markdown("---")
    st.markdown("### 🎬 Popular Movies to Try")
    popular_movies = df.nlargest(10, 'Popularity')[['Title', 'Release_Date', 'Genre', 'Popularity']]
    st.dataframe(popular_movies, use_container_width=True, hide_index=True)


