import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

RATINGS_PATH = os.path.join(os.path.dirname(__file__), "ratings.csv")
MOVIES_PATH = os.path.join(os.path.dirname(__file__), "movies.csv")

_FALLBACK_MOVIES = {
    1: "The Shawshank Redemption",
    2: "The Godfather",
    3: "The Dark Knight",
    4: "Pulp Fiction",
    5: "Forrest Gump",
    6: "Inception",
    7: "Fight Club",
    8: "The Matrix",
    9: "Interstellar",
    10: "Parasite",
    11: "Gladiator",
    12: "Titanic",
}

_FALLBACK_RATINGS = [
    (1, 1, 5), (1, 2, 4), (1, 3, 4), (1, 5, 5), (1, 6, 3),
    (2, 1, 4), (2, 2, 5), (2, 4, 3), (2, 6, 4), (2, 7, 5),
    (3, 3, 5), (3, 6, 5), (3, 8, 4), (3, 9, 5), (3, 11, 2),
    (4, 1, 5), (4, 5, 4), (4, 10, 5), (4, 12, 4), (4, 2, 3),
    (5, 2, 4), (5, 4, 5), (5, 7, 4), (5, 8, 5), (5, 6, 4),
    (6, 3, 4), (6, 6, 5), (6, 9, 5), (6, 8, 4), (6, 1, 3),
    (7, 5, 5), (7, 10, 4), (7, 12, 5), (7, 2, 4), (7, 1, 4),
    (8, 4, 4), (8, 7, 5), (8, 8, 5), (8, 3, 4), (8, 6, 4),
    (9, 9, 5), (9, 6, 4), (9, 3, 5), (9, 11, 3), (9, 8, 4),
    (10, 10, 5), (10, 12, 4), (10, 5, 5), (10, 1, 4), (10, 2, 4),
]

if os.path.exists(RATINGS_PATH):
    ratings_df = pd.read_csv(RATINGS_PATH)
else:
    ratings_df = pd.DataFrame(_FALLBACK_RATINGS, columns=["userId", "movieId", "rating"])

if os.path.exists(MOVIES_PATH):
    MOVIES = pd.read_csv(MOVIES_PATH).set_index("movieId")["title"].to_dict()
else:
    MOVIES = _FALLBACK_MOVIES


def build_user_item_matrix(df):
    matrix = df.pivot_table(index="userId", columns="movieId", values="rating")
    return matrix


def compute_user_similarity(matrix):
    filled = matrix.fillna(0)
    sim = cosine_similarity(filled)
    return pd.DataFrame(sim, index=matrix.index, columns=matrix.index)


def predict_rating(user_id, movie_id, matrix, similarity):
    if movie_id not in matrix.columns:
        return np.nan

    movie_ratings = matrix[movie_id]
    rated_mask = movie_ratings.notna()

    if rated_mask.sum() == 0:
        return np.nan

    sims = similarity.loc[user_id, rated_mask.index[rated_mask]]
    ratings = movie_ratings[rated_mask]

    if sims.abs().sum() == 0:
        return ratings.mean()

    return np.dot(sims, ratings) / sims.abs().sum()


def recommend_top_n(user_id, matrix, similarity, n=5):
    seen = matrix.loc[user_id].dropna().index.tolist()
    unseen = [m for m in matrix.columns if m not in seen]

    predictions = {
        m: predict_rating(user_id, m, matrix, similarity)
        for m in unseen
    }

    predictions = {
        m: r for m, r in predictions.items()
        if not np.isnan(r)
    }

    top = sorted(
        predictions.items(),
        key=lambda x: x[1],
        reverse=True
    )[:n]

    return [
        (MOVIES.get(m, f"Movie {m}"), round(r, 2))
        for m, r in top
    ]


def evaluate_rmse(df, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )

    train_matrix = build_user_item_matrix(train_df)
    train_similarity = compute_user_similarity(train_matrix)

    actuals = []
    preds = []

    for _, row in test_df.iterrows():
        u, m, r = row["userId"], row["movieId"], row["rating"]

        if u not in train_matrix.index:
            continue

        pred = predict_rating(u, m, train_matrix, train_similarity)

        if np.isnan(pred):
            continue

        actuals.append(r)
        preds.append(pred)

    if not actuals:
        return None

    rmse = np.sqrt(mean_squared_error(actuals, preds))
    return rmse


def main():
    matrix = build_user_item_matrix(ratings_df)
    similarity = compute_user_similarity(matrix)

    print("=" * 55)
    print("MOVIE RECOMMENDATION SYSTEM (User-Based CF)")
    print("=" * 55)

    rmse = evaluate_rmse(ratings_df)

    if rmse is not None:
        print(f"\nModel Evaluation -> RMSE on held-out ratings: {rmse:.3f}")
    else:
        print("\nNot enough held-out data to compute RMSE.")

    print(f"\nAvailable user IDs: {list(matrix.index)}")

    selected_user = 3

    print(f"\nTop 5 movie recommendations for User {selected_user}:")

    recs = recommend_top_n(
        selected_user,
        matrix,
        similarity,
        n=5
    )

    for i, (title, score) in enumerate(recs, start=1):
        print(f"  {i}. {title}  (predicted rating: {score})")


if __name__ == "__main__":
    main()