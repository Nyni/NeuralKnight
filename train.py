import preprocess
import AI
from torch.utils.data import DataLoader
from torch import nn
import torch

games = preprocess.parse_pgn('libchess_db.pgn', 2000, 100)
data_train = preprocess.ChessDataset(games)
data_loader = DataLoader(data_train, batch_size=32, drop_last=True)

model = AI.ChessCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)

def train_one_epoch():
    running_loss = 0
    last_loss = 0
    
    print("Max data batches:", len(data_loader))
    for i, data in enumerate(data_loader):
        input, output = data
        print("Training data: ", i)

        optimizer.zero_grad()

        prediction = model(input)

        loss = loss_fn(prediction, output) 
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        print("lost:" , loss.item())
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch

    return last_loss

print("loss:", train_one_epoch())
# print("Max data:", len(data_loader))
