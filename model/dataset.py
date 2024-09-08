import torch
from torch.utils.data import Dataset


class GameDataset(Dataset):
    def __init__(self, users, items, hours):
        self.users = users
        self.items = items
        self.hours = hours
        
    # len(movie_dataset)
    def __len__(self): # Number of Users
        return len(self.users)

    # movie_dataset[1]
    def __getitem__(self, idx):

        users = self.users[idx]
        items = self.items[idx]
        hours = self.hours[idx]

        return {
            "user_id" : torch.tensor(users, dtype=torch.long),
            "title" : torch.tensor(items, dtype=torch.long),
            "hours" : torch.tensor(hours, dtype=torch.float)
        }