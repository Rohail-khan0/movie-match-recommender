# Movie Recommendation System
A Python-based movie recommendation engine utilizing TF-IDF Vectorization and Cosine Similarity. Features include comprehensive data cleaning, exploratory data analysis (EDA), and a genre-matching precision evaluation metric.
## 🚀 Overview
The system recommends movies similar to a user-provided title by processing textual "tags." It transforms qualitative descriptions into mathematical vectors using Natural Language Processing (NLP) techniques, allowing for fast and accurate similarity computations across a dataset of nearly 10,000 films.

## 🛠️ Features
- **Data Preprocessing**: Advanced cleaning including timestamp conversion to years and handling of missing values.
- **Feature Engineering**: Creation of a consolidated "tags" column combining movie overviews and genres to capture the essence of each film.
- **NLP Pipeline**: 
    - **TF-IDF Vectorization**: Converts text into meaningful numerical features while penalizing overly common words.
    - **Cosine Similarity**: Measures the distance between vectors to find the closest thematic matches.
- **Evaluation Metric**: Implements a custom `precision_at_k` function to validate recommendations based on genre relevance.

## 📊 Dataset Insight
The model utilizes a dataset of **9,827 movies** with features including:
- **Metadata**: Title, Overview, Genre, Original Language.
- **Metrics**: Popularity, Vote Average, and Vote Count.
- **Categorization**: Films are classified into segments like "Popular," "Average," and "Not Popular" based on their statistical distributions.

## 💻 Technical Stack
- **Languages**: Python
- **Libraries**: 
    - `Pandas` & `NumPy` for data manipulation.
    - `Scikit-Learn` for NLP and similarity algorithms.
    - `Matplotlib` & `Seaborn` for data visualization.

## 📈 Model Performance
Testing on the "Avengers" franchise demonstrated high accuracy, achieving a **Precision@5 score of 1.0** (meaning 100% of the top 5 recommendations shared at least one genre with the original query).

## ⚙️ Setup & Usage
1. **Clone the repo**:
   ```bash
   git clone [https://github.com/yourusername/movie-match-recommender.git](https://github.com/yourusername/movie-match-recommender.git)

2. **Install dependencies:**
- pip install pandas numpy scikit-learn matplotlib seaborn streamlit

3. **Run the Streamlit App:**
- streamlit run Movie_Recommendation_System.py

## 🗺️ Future Roadmap
[ ] Integrate movie poster visualization via TMDB API.
[ ] Implement Collaborative Filtering to account for user rating behavior.
[ ] Deploy the app using Streamlit Cloud or Heroku.

### Key Summary of Workflow (Internal Reference)
* **Step 1: Data Cleaning** - Dropped non-essential columns like `Poster_Url` and converted release dates.
* **Step 2: EDA** - Visualized movie counts per popularity category.
* **Step 3: Vectorization** - Calculated TF-IDF matrices on overviews and genres.
* **Step 4: Inference** - Mapped similarity scores back to movie titles for user output.
