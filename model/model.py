import torch


class RecSysModel(torch.nn.Module):
    def __init__(self, n_users, n_items, embedding_size=64, hidden_dim=128, dropout_rate=0.2):
        super().__init__()

        #create embeddings
        self.user_embed = torch.nn.Embedding(num_embeddings=n_users, embedding_dim=embedding_size)
        self.item_embed = torch.nn.Embedding(num_embeddings=n_items, embedding_dim=embedding_size)


        # hidden layers
        self.fc1 = torch.nn.Linear(2 * embedding_size, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

        self.dropout = torch.nn.Dropout(p=dropout_rate)

        self.relu = torch.nn.ReLU()
        
    def forward(self, users, items, hours=None):
        user_embeds = self.user_embed(users)
        item_embeds = self.item_embed(items)
        
        output = torch.cat([user_embeds, item_embeds], dim=1)

        x = self.relu(self.fc1(output))
        x = self.dropout(x)
        output = self.fc2(x)
        return output
    
def load_model(model_path, device, n_users, n_items):
    model = RecSysModel(n_users, n_items)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()
    return model