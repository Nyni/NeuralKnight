import preprocess
import AI
from torch.utils.data import DataLoader
from torch import nn
import torch

MIN_ELO = 2000
MAX_TRAINING_GAMES = 1000

def split_data(data, train_percentage):
    from random import shuffle
    shuffle(data)
    ceil = int(len(data) * train_percentage/100)
    return (data[:ceil], data[ceil+1:])

games = preprocess.parse_pgn('libchess_db.pgn', MIN_ELO, MAX_TRAINING_GAMES)
training_games, testing_games = split_data(games, 70)

training_data = preprocess.ChessDataset(training_games)
training_data_loader = DataLoader(training_data, batch_size=32, drop_last=True)

testing_data = preprocess.ChessDataset(testing_games)
testing_data_loader = DataLoader(testing_data, batch_size=32, drop_last=True)

model = AI.ChessCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)

print("Training ChessCNN")
print("Training data:", len(training_games))
print("Testing data:", len(testing_games))
def train_one_epoch():
    running_loss = 0
    last_loss = 0
    
    max_batches = len(training_data_loader)
    for i, data in enumerate(training_data_loader):
        input, output = data

        optimizer.zero_grad()

        prediction = model(input)

        loss = loss_fn(prediction, output) 
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        print(f"batch {i + 1} loss:" , loss.item())
        if i % max_batches == max_batches - 1:
            last_loss = running_loss / max_batches # loss per batch

    return last_loss

def train_epoch(num = 1):
    best_vloss = 100_000
    max_batches = len(testing_data_loader)
    for epoch in range(1, num):
        print("Epoch:", epoch)
        model.train()
        avg_loss = train_one_epoch()

        running_vloss = 0.0
        # testing the model
        model.eval()
        
        with torch.no_grad():
            for vdata in testing_data_loader:
                vinput, voutput = vdata
                pred = model(vinput)
                vloss = loss_fn(pred, voutput)
                running_vloss += vloss

        avg_vloss = running_vloss / max_batches
        print(f"Training loss: {avg_loss}; Validation loss: {avg_vloss}")

        if best_vloss > avg_vloss:
            best_vloss = avg_vloss
            torch.save(model.state_dict(), f"ChessCNN_E{epoch}_L_{best_vloss:.5E}.pt")

train_epoch()
