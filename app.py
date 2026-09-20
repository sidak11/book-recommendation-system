import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "svd_model.pkl")
books = joblib.load(BASE_DIR / "book_data.pkl")
ratings = joblib.load(BASE_DIR / "ratings_data.pkl")


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📚 Book Recommendation System")

st.write(
    "Rate 5 books and our collaborative filtering model "
    "will recommend books personalized to your preferences."
)

st.divider()


# --------------------------------------------------
# PREPARE POPULAR BOOKS FOR SELECTION
# --------------------------------------------------

book_counts = ratings["ISBN"].value_counts()

popular_books = (
    book_counts
    .head(1000)
    .index
)

selection_books = books[
    books["ISBN"].isin(popular_books)
].drop_duplicates("ISBN").copy()

selection_books["Display"] = (
    selection_books["Book-Title"].astype(str)
    + " — "
    + selection_books["Book-Author"].fillna("Unknown").astype(str)
)

selection_books = selection_books.sort_values("Book-Title")

book_options = selection_books[
    ["ISBN", "Display"]
].values.tolist()

book_dict = dict(book_options)


# --------------------------------------------------
# USER RATING SECTION
# --------------------------------------------------

st.header("⭐ Rate 5 Books")

st.write(
    "Choose five books you have read and rate each one from 1 to 10."
)

ratings_input = []

for i in range(5):

    col1, col2 = st.columns([4, 1])

    with col1:
        selected_display = st.selectbox(
            f"Book {i + 1}",
            ["Select a book"] + list(book_dict.values()),
            key=f"book_{i}"
        )

    with col2:
        rating_value = st.slider(
            f"Rating {i + 1}",
            min_value=1,
            max_value=10,
            value=8,
            key=f"rating_{i}"
        )

    ratings_input.append(
        (selected_display, rating_value)
    )


# --------------------------------------------------
# RECOMMENDATION FUNCTION
# --------------------------------------------------

def recommend_for_new_user(user_ratings, model, books_df, ratings_df, n=5):

    selected_isbns = []

    for display_name, rating in user_ratings:

        if display_name == "Select a book":
            continue

        isbn = selection_books.loc[
            selection_books["Display"] == display_name,
            "ISBN"
        ].iloc[0]

        selected_isbns.append((isbn, rating))

    if len(selected_isbns) < 5:
        return None

    # SVD components
    q = model.qi
    bu = model.bi
    mu = model.trainset.global_mean

    # Build matrices from the five rated books
    X = []
    y = []

    for isbn, rating in selected_isbns:

        try:
            inner_iid = model.trainset.to_inner_iid(isbn)

            item_vector = q[inner_iid]

            X.append(item_vector)

            # Remove global mean and item bias
            adjusted_rating = (
                rating
                - mu
                - bu[inner_iid]
            )

            y.append(adjusted_rating)

        except ValueError:
            continue

    if len(X) < 3:
        return None

    X = np.array(X)
    y = np.array(y)

    # Infer a temporary user latent vector
    # using the five ratings.
    regularization = 0.1

    identity = np.eye(X.shape[1])

    user_vector = np.linalg.solve(
        X.T @ X + regularization * identity,
        X.T @ y
    )

    # Candidate books
    rated_isbns = {isbn for isbn, _ in selected_isbns}

    candidates = books_df[
        ~books_df["ISBN"].isin(rated_isbns)
    ].drop_duplicates("ISBN")

    predictions = []

    for isbn in candidates["ISBN"]:

        try:
            inner_iid = model.trainset.to_inner_iid(isbn)

            item_vector = model.qi[inner_iid]
            item_bias = model.bi[inner_iid]

            predicted_rating = (
                mu
                + item_bias
                + np.dot(user_vector, item_vector)
            )

            predicted_rating = np.clip(
                predicted_rating,
                1,
                10
            )

            predictions.append(
                (isbn, predicted_rating)
            )

        except ValueError:
            continue

    predictions.sort(
        key=lambda x: x[1],
        reverse=True
    )

    top_predictions = predictions[:n]

    result = pd.DataFrame(
        top_predictions,
        columns=["ISBN", "Predicted Rating"]
    )

    result = result.merge(
        books_df[
            ["ISBN", "Book-Title", "Book-Author"]
        ],
        on="ISBN",
        how="left"
    )

    return result[
        [
            "Book-Title",
            "Book-Author",
            "Predicted Rating"
        ]
    ]


# --------------------------------------------------
# RECOMMEND BUTTON
# --------------------------------------------------

if st.button(
    "✨ Get My Recommendations",
    use_container_width=True
):

    with st.spinner("Analyzing your preferences..."):

        recommendations = recommend_for_new_user(
            ratings_input,
            model,
            books,
            ratings,
            n=5
        )

    if recommendations is None:

        st.warning(
            "Please select 5 different books that are available "
            "in the recommendation model."
        )

    else:

        st.divider()

        st.header("🎯 Your Personalized Recommendations")

        for index, row in recommendations.iterrows():

            st.subheader(
                f"{index + 1}. {row['Book-Title']}"
            )

            st.write(
                f"✍️ Author: {row['Book-Author']}"
            )

            st.write(
                f"⭐ Predicted Rating: "
                f"{row['Predicted Rating']:.2f}/10"
            )

            st.divider()


# --------------------------------------------------
# HOW IT WORKS
# --------------------------------------------------

st.header("🧠 How It Works")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1️⃣ User Input")
    st.write(
        "The user rates five books they have already read."
    )

with col2:
    st.subheader("2️⃣ SVD")
    st.write(
        "Collaborative filtering learns hidden preferences "
        "from user-book rating patterns."
    )

with col3:
    st.subheader("3️⃣ Recommendation")
    st.write(
        "The system predicts ratings for unseen books "
        "and returns the top five."
    )


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.divider()

st.header("📊 Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Algorithm", "SVD")

with col2:
    st.metric("Latent Factors", "50")

with col3:
    st.metric("Test RMSE", "1.59")