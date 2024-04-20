import preprocess
import AI
from torch.utils.data import DataLoader
from torch import nn
import torch

MIN_ELO = 2000
MAX_TRAINING_GAMES = 5000

games = preprocess.parse_pgn('libchess_db.pgn', MIN_ELO, MAX_TRAINING_GAMES)
data_train = preprocess.ChessDataset(games)
data_loader = DataLoader(data_train, batch_size=32, drop_last=True)

model = AI.ChessCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)

def train_one_epoch():
    running_loss = 0
    last_loss = 0
    
    max_batches = len(data_loader)
    for i, data in enumerate(data_loader):
        input, output = data

        optimizer.zero_grad()

        prediction = model(input)

        loss = loss_fn(prediction, output) 
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        print("loss:" , loss.item())
        if i % max_batches == max_batches - 1:
            last_loss = running_loss / max_batches # loss per batch

    return last_loss

def train_epoch(num = 1):
    model.train()

    for i in range(num):
        print("Epoch:", i + 1)
        print("loss per epoch:", train_one_epoch())

    torch.save(model.state_dict(), "chess_model.pt")

train_epoch()
