# 📚 Book Recommendation System

A Machine Learning recommendation system that recommends books to users based on their past ratings.

The project uses **Collaborative Filtering with SVD (Singular Value Decomposition)** from the Surprise library. A Streamlit application allows a new user to rate five books and receive five personalized book recommendations.

---

## 🚀 Project Objective

The goal of this project is to build a personalized book recommendation system using historical user-book rating interactions.

The system:

1. Loads users, books, and ratings data.
2. Uses explicit ratings for recommendation modeling.
3. Reduces sparsity by filtering users and books with very few interactions.
4. Trains an SVD collaborative filtering model.
5. Tunes important SVD hyperparameters.
6. Evaluates the model using RMSE.
7. Generates Top-5 personalized recommendations.
8. Provides a Streamlit interface for new users.

---

## 📊 Dataset

The project uses the **Book Recommendation Dataset / Book-Crossing Dataset**.

Dataset source:

https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset

The dataset contains three main files:

```text
Users.csv
Books.csv
Ratings.csv
```

### Users

Contains information about users:

- User-ID
- Location
- Age

### Books

Contains book information:

- ISBN
- Book-Title
- Book-Author
- Year-Of-Publication
- Publisher
- Image URLs

### Ratings

Contains user-book interactions:

- User-ID
- ISBN
- Book-Rating

---

## 🔍 Explicit vs Implicit Ratings

The dataset contains ratings from **0 to 10**.

```text
0       → Implicit interaction
1–10    → Explicit rating
```

For the SVD recommendation model, only explicit ratings were used:

```python
explicit_ratings = ratings[ratings["Book-Rating"] > 0]
```

This allows the model to learn from actual user rating preferences.

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

### 1. Missing Value Analysis

Missing values were checked in all three datasets.

The `Age` column contains many missing values, but age is not required for the collaborative filtering model.

### 2. Explicit Rating Selection

Zero ratings were removed and only ratings from 1 to 10 were retained.

### 3. Sparse Data Reduction

To reduce the sparsity of the user-item matrix:

- Users with fewer than 5 ratings were removed.
- Books with fewer than 5 ratings were removed.

### 4. Dataset Joining

Ratings were connected with book information using:

```text
Ratings.ISBN → Books.ISBN
```

Users can also be connected through:

```text
Ratings.User-ID → Users.User-ID
```

The main recommendation model uses:

```text
User-ID + ISBN + Book-Rating
```

---

## 🤖 Recommendation Algorithm

### Collaborative Filtering

The system uses **model-based collaborative filtering**.

The main algorithm is:

**SVD — Singular Value Decomposition**

SVD learns hidden/latent relationships between:

```text
Users ↔ Books
```

It learns latent factors representing user preferences and book characteristics.

genui{"learning_viz":{"type_id":"SINGULAR_VALUE_DECOMPOSITION","initial_values":{"rank":2}}}

The trained model predicts how a user might rate a book they have not rated before.

---

## ⚙️ Model Training

The ratings were divided into:

```text
80% → Training
20% → Testing
```

The SVD model was trained using the **Surprise** library.

### Initial Model

The initial SVD configuration used:

```text
n_factors = 100
n_epochs = 20
learning rate = 0.005
regularization = 0.02
```

---

## 🔧 Hyperparameter Tuning

A small GridSearchCV search was performed to tune important parameters.

Parameters tested included:

```text
n_factors:
50, 100

n_epochs:
20

lr_all:
0.005, 0.01

reg_all:
0.02
```

### Best Parameters

```text
n_factors = 50
n_epochs = 20
lr_all = 0.005
reg_all = 0.02
```

---

## 📈 Model Evaluation

The model was evaluated using **Root Mean Square Error (RMSE)**.

### Results

| Metric | Result |
|---|---:|
| Best Cross-Validation RMSE | 1.5872 |
| Final Test RMSE | **1.5939** |

A lower RMSE indicates that the predicted ratings are closer to the actual ratings.

---

## 🎯 Top-5 Recommendation

After training, the system predicts ratings for books that the user has not rated.

The books are sorted by predicted rating and the top five are returned.

Example:

```text
1. Recommended Book A — 8.72
2. Recommended Book B — 8.54
3. Recommended Book C — 8.41
4. Recommended Book D — 8.29
5. Recommended Book E — 8.17
```

The actual recommendations depend on the user's ratings.

---

## 🌟 Innovative Feature: New User Recommendations

The Streamlit application addresses the **cold-start problem**.

A completely new user has no previous rating history, so the application asks the user to rate five books.

```text
New User
   ↓
Rate 5 Books
   ↓
Estimate User Preferences
   ↓
Compare With Learned Book Factors
   ↓
Predict Unseen Books
   ↓
Top 5 Recommendations
```

This provides personalized recommendations without requiring the user to already exist in the training dataset.

---

## 🌐 Streamlit Application

The project includes an interactive Streamlit application.

### User Flow

```text
1. Open the application
          ↓
2. Select 5 books
          ↓
3. Give each book a rating from 1–10
          ↓
4. Click "Get My Recommendations"
          ↓
5. System analyzes preferences
          ↓
6. Top 5 books are displayed
```

### Application Features

- 📚 Book selection
- ⭐ Rating from 1–10
- 🧠 SVD-based collaborative filtering
- 🎯 Personalized Top-5 recommendations
- 📊 Predicted rating for each recommendation
- ❄️ New-user cold-start handling
- 💻 Simple interactive interface

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Matplotlib | Visualization |
| Seaborn | Data visualization |
| Scikit-Surprise | Recommendation algorithm |
| SVD | Collaborative filtering |
| Joblib | Model serialization |
| Streamlit | Web application |
| Jupyter Notebook | Model development |
| Git & GitHub | Version control |

---

## 📁 Project Structure

```text
Task3-Book-Recommendation/
│
├── app.py
├── book_recommendation.ipynb
├── svd_model.pkl
├── book_data.pkl
├── ratings_data.pkl
├── requirements.txt
├── README.md
├── .gitignore
│
├── Books.csv
├── Ratings.csv
└── Users.csv
```

### Important Files

**`book_recommendation.ipynb`**  
Contains data analysis, preprocessing, SVD training, hyperparameter tuning, evaluation, and recommendation experiments.

**`app.py`**  
Streamlit application for collecting five book ratings and generating personalized recommendations.

**`svd_model.pkl`**  
Saved trained SVD model.

**`book_data.pkl`**  
Saved book information used by the recommendation system.

**`ratings_data.pkl`**  
Filtered rating data used by the application.

**`requirements.txt`**  
Python dependencies required to run the project.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Task3-Book-Recommendation
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the environment

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

---

## 📦 Requirements

The application requires:

```text
streamlit
pandas
numpy
scikit-surprise
joblib
```

---

## 🧪 Example

A new user might provide:

```text
The Hobbit             → 9/10
Harry Potter            → 10/10
The Alchemist           → 8/10
1984                    → 9/10
The Da Vinci Code      → 7/10
```

The system uses these ratings to estimate the user's preferences and produces five unseen book recommendations.

---

## ❄️ Cold-Start Problem

### What is the cold-start problem?

A recommendation system cannot easily personalize recommendations for a user who has no previous interactions.

### Solution used in this project

The application asks a new user to provide five initial ratings.

These ratings are used to estimate a temporary user preference representation using the learned SVD item factors.

This allows the system to provide personalized recommendations without requiring historical data for that user.

### Future improvements

A production system could improve cold-start handling with:

- Popular books for initial onboarding
- Genre-based questions
- Author preferences
- Content-based recommendations
- Hybrid collaborative + content-based filtering
- Gradual personalization as the user rates more books

---

## 🎯 Learning Outcomes

This project demonstrates practical understanding of:

- Recommendation systems
- Collaborative filtering
- SVD
- Latent factor models
- User-item interactions
- Sparse datasets
- Explicit and implicit feedback
- Hyperparameter tuning
- RMSE evaluation
- Top-N recommendations
- Cold-start problem
- Model serialization
- Streamlit deployment

---

## 🔮 Future Improvements

Possible improvements include:

- Hybrid recommendation system
- Content-based book recommendations
- Genre-aware recommendations
- Author similarity
- Book cover images
- User accounts and recommendation history
- Improved cold-start strategies
- More advanced matrix factorization methods
- Neural collaborative filtering
- Larger recommendation datasets

---

## 👨‍💻 Project Summary

**Project:** Book Recommendation System

**Domain:** Machine Learning / Recommendation Systems

**Algorithm:** SVD Collaborative Filtering

**Final Test RMSE:** 1.5939

**Application:** Streamlit

**Dataset:** Book-Crossing / Book Recommendation Dataset

**Recommendation Output:** Top 5 personalized books

**Cold-Start Strategy:** Five initial user ratings
