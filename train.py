import torch
from torch import nn
from tqdm import tqdm
from models.CustomNet import CustomNet
from eval import validate
from dataset.Preprocess import Preprocess
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize wandb
wandb.init(project="tiny-imagenet", entity="your-entity")

# Define train function
def train(epoch, model, train_loader, criterion, optimizer):
  model.train()
  running_loss = 0.0
  correct = 0
  total = 0

  for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
    data, target = data.to(device), target.to(device)

    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    running_loss += loss.item() * data.size(0)
    _, predicted = output.max(1)
    total += target.size(0)
    correct += predicted.eq(target).sum().item()

  train_loss = running_loss / len(train_loader.dataset)
  train_accuracy = 100. * correct / total
  print(f'Train Epoch: {epoch}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy}')

def main():
    model = CustomNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.3)
    criterion = nn.CrossEntropyLoss()
    epochs = 10
    best_accuracy = 0
    patience = 2
    counter = 0

    train_loader, val_loader = Preprocess()

    for epoch in range(epochs):
        train(epoch, model, train_loader, criterion, optimizer)
        val_accuracy = validate(model, val_loader, criterion)
        scheduler.step()
        wandb.log({"acc": val_accuracy})

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break
    wandb.finish()

    print(f"Best accuracy : {best_accuracy}")

if __name__ == '__main__':
    main()