import pandas as pd
import torch
from sklearn import model_selection
from torch.utils.data import DataLoader
from model.model import RecSysModel
from model.train import train
from model.dataset import GameDataset
from model.data import load_and_preprocess_data
import os

print("current working directory:", os.getcwd())
script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the absolute paths
reviews_path = os.path.join(script_dir, "data/steam-200k.csv")
test_ratings_path = os.path.join(script_dir, "data/test_reviews.csv")

MODEL_PATH = os.path.join(script_dir, "models/saved_model.pth")
BATCH_SIZE = 32
EPOCHS = 2

def main():
    (
        df,
        user_encoder, 
        item_encoder
    ) = load_and_preprocess_data(
        reviews_path
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    n_users = df['user_id'].nunique()
    n_items = df['title'].nunique()

    df['user_id'] = user_encoder.fit_transform(df['user_id'].values)
    df['title'] = item_encoder.fit_transform(df['title'].values)

    df_train, df_valid = model_selection.train_test_split(
        df, test_size=0.1, random_state=3
    )

    df_valid.to_csv(test_ratings_path, index=False) #

    train_dataset = GameDataset(
        users = df_train['user_id'].values,
        items = df_train['title'].values,
        hours = df_train['hours'].values,
        
    )

    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True
    )

    recommendation_model = RecSysModel(n_users, n_items, embedding_size=64, hidden_dim=128, dropout_rate=0.1)

    optimizer = torch.optim.Adam(recommendation_model.parameters()) #gradient descent
    sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
    loss_fn = torch.nn.MSELoss()

    train(recommendation_model.to(device),
        train_dataset, 
        train_loader, 
        loss_fn, 
        optimizer, 
        model_path=MODEL_PATH, 
        device=device,
        epochs=EPOCHS)
    

if __name__ == "__main__":
    main()
