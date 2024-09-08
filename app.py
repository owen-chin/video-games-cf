# app.py
from flask import Flask, request, render_template
import pandas as pd
import torch
from model.data import load_and_preprocess_data
from model.model import load_model
from model.model import RecSysModel

app = Flask(__name__)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
reviews_path = "data/steam-200k.csv"
# Load and preprocess data
(
    df,
    user_encoder, 
    item_encoder
) = load_and_preprocess_data(
    reviews_path
)

# Initialize model
n_users = df['user_id'].nunique()
n_items = df['title'].nunique()

model_path = 'models/saved_model.pth'  # Ensure this path is correct and matches the path in train_model.py
recommendation_model = load_model(model_path, device, n_users, n_items)

# Label encoding for movie ids
lbl_item = item_encoder

# All games
all_games = df['title'].unique().tolist()

# Function to get top recommendations
def top_recommendations(user_id, all_titles, k=5, batch_size=100):
    recommendation_model.eval()

    played_titles = set(df[df['user_id'] == user_id]['title'].tolist())
    unplayed_titles = [m for m in all_titles if m not in played_titles]

    prediction = []
    top_k_recommendations = []

    with torch.no_grad():
        for i in range(0, len(unplayed_titles), batch_size):
            batched_unwatched = unplayed_titles[i:i+batch_size]
            title_tensor = torch.tensor(batched_unwatched).to(device)
            user_tensor = torch.tensor([user_id] * len(batched_unwatched)).to(device)
            prediction_model = recommendation_model(user_tensor, title_tensor).view(-1).tolist()
            prediction.extend(zip(batched_unwatched, prediction_model))

    prediction.sort(key=lambda x: x[1], reverse=True)

    for (m_id, _) in prediction[:k]:
        top_k_recommendations.append(m_id)

    # Convert this encoded movieId's back to their original ids
    top_k_recommendations = lbl_item.inverse_transform(top_k_recommendations)
    
    return top_k_recommendations

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_id = int(request.form['user_id'])
        if user_id not in df['user_id'].values:
            raise ValueError(f'User ID {user_id} does not exist')
        recommendations = top_recommendations(user_id, all_games, k=5).tolist()
        
        return render_template('index.html', recommendations=recommendations)
    except ValueError as e:
        app.logger.error(f"Error: {e}")

        return render_template('index.html', error=str(e))
    except Exception as e:
        app.logger.error(f"An unexpected error occurred: {e}")

        return render_template("index.html", error=str(e))
    
if __name__ == '__main__':
    app.run(host="127.0.0.1", debug=True)
