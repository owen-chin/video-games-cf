import torch
import pandas as pd
from torch.utils.data import DataLoader
from model.model import RecSysModel
from model.dataset import GameDataset
from model.data import load_and_preprocess_data
from sklearn.metrics import root_mean_squared_error
from collections import defaultdict
import numpy as np

#define file paths
reviews_path = "data/steam-200k.csv"
test_ratings_path = "data/test_reviews.csv"
MODEL_PATH = "./models/saved_model.pth"
BATCH_SIZE = 32

# Load test data
df_valid = pd.read_csv(test_ratings_path)

# Load and preprocess data
(
    df,
    user_encoder, 
    item_encoder
) = load_and_preprocess_data(
    reviews_path
)

valid_dataset = GameDataset(
    users = df_valid['user_id'].values,
    items = df_valid['title'].values,
    hours = df_valid['hours'].values,
)

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False
)

# Initialize device and model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

recommendation_model = RecSysModel(
    n_users=df['user_id'].nunique(),
    n_items=df['title'].nunique(),
    embedding_size=64,
    hidden_dim=128,
    dropout_rate=0.1
)
recommendation_model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
recommendation_model.to(device)

def calculate_precision_recall(user_ratings, k, threshold):
    user_ratings.sort(key=lambda x: x[0], reverse=True)
    n_rel = sum(true_r >= threshold for _, true_r in user_ratings)
    n_rec_k = sum(est >= threshold for est, _ in user_ratings[:k])
    n_rel_and_rec_k = sum(
        (true_r >= threshold) and (est >= threshold) for est, true_r in user_ratings[:k]
    )

    precision = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
    recall = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
    return precision, recall

def find_RMSE():
    # Root Mean Squared Error
    y_true = []
    y_pred = []

    recommendation_model.eval()

    with torch.no_grad():
        for i, valid_data in enumerate(valid_loader):
            model_output = recommendation_model(valid_data['user_id'].to(device), valid_data['title'].to(device))

            hours = valid_data['hours'].to(device)
            y_true.extend(hours.cpu().numpy()) 
            y_pred.extend(model_output.cpu().numpy())


    # actually calc RMSE
    rmse = root_mean_squared_error(y_true, y_pred)
    print(f"RMSE: {rmse:.4f}")


def main():
    find_RMSE()

    user_hours_comparison = defaultdict(list)

    with torch.no_grad():
        for valid_data in valid_loader:
            users = valid_data["user_id"].to(device)
            titles = valid_data["title"].to(device)
            hours = valid_data["hours"].to(device)
            output = recommendation_model(users, titles)

            for user, pred, true in zip(users, output, hours):
                user_hours_comparison[user.item()].append((pred[0].item(), true.item()))

    user_precisions = dict()
    user_based_recalls = dict()

    k = 50
    threshold = 3

    for user_id, user_hours in user_hours_comparison.items():
        precision, recall = calculate_precision_recall(user_hours, k, threshold)
        user_precisions[user_id] = precision
        user_based_recalls[user_id] = recall


        average_precision = sum(prec for prec in user_precisions.values()) / len(user_precisions)
        average_recall = sum(rec for rec in user_based_recalls.values()) / len(user_based_recalls)

    print(f"precision @ {k}: {average_precision:.4f}")
    print(f"recall @ {k}: {average_recall:.4f}")

if __name__ == "__main__":
    main()
