import streamlit as st
import pandas as pd
import joblib
import difflib
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64

# Function to set a background image
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Call the function with your image file
set_background(r"C:\Users\TECH POINT\Desktop\Book Recommender\Background.jpg")  # Replace with your image path


# Load precomputed models
df = pd.read_csv(r"C:\Users\TECH POINT\Desktop\Book Recommender\data.csv")
similarity = joblib.load("Books_recommender.pkl")
vectorizer = TfidfVectorizer()

# Preprocess data
selected_features = ['title', 'authors', 'categories', 'published_year']
for feature in selected_features:
    df[feature] = df[feature].fillna('')
combined_features = df['title'] + ' ' + df['categories'] + ' ' + df['authors'] + ' ' + df['published_year'].astype(str)
feature_vectors = vectorizer.fit_transform(combined_features)

# Streamlit UI
st.title("📚 Book Recommendation System")
st.write("Enter a book name to get recommendations!")



# User input
book_name = st.text_input("Enter Book Title", placeholder = "Enter here!!")

if st.button("Find Recommendations"):
    list_of_all_titles = df['title'].tolist()
    find_close_match = difflib.get_close_matches(book_name, list_of_all_titles)
    
    if find_close_match:
        close_match = find_close_match[0]
        index_of_the_book = df[df.title == close_match].index[0]
        similarity_score = list(enumerate(similarity[index_of_the_book]))
        sorted_similar_books = sorted(similarity_score, key=lambda x: x[1], reverse=True)
        
        st.subheader("Top 5 Similar Books:")
        for i, book in enumerate(sorted_similar_books[:5], 1):
            index = book[0]
            title_from_index = df.iloc[index]['title'] 
            st.write(f"{i}. {title_from_index}")
    else:
        st.error("No close match found. Try another book title!")

with st.container(height=300):
    # Display random book titles
    st.subheader("📖 Explore Some Books:")
    random_books = random.sample(df['title'].tolist(), min(20, len(df)))
    for book in random_books:
        st.code(f"- {book}", language= None)



