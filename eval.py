import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define validate function
def validate(model, val_loader, criterion):
  model.eval()
  running_loss = 0.0
  correct = 0
  total = 0

  with torch.no_grad():
    for batch_idx, (data, target) in enumerate(val_loader):
      data, target = data.to(device), target.to(device)

      output = model(data)
      loss = criterion(output, target)
      running_loss += loss.item()
      _, predicted = output.max(1)
      total += target.size(0)
      correct += predicted.eq(target).sum().item()

  val_loss = running_loss / len(val_loader)
  val_accuracy = 100. * correct / total
  print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy}')
  return val_accuracy
