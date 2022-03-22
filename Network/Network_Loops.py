import torch
from torch import reshape

def one_hot_vector_labels(scalar_labels):
  vector_labels = torch.empty((len(scalar_labels),2))
  for index in range(len(scalar_labels)):
    if scalar_labels[index] == 0:
      vector_labels[index,0] = 1
      vector_labels[index,1] = 0
    elif scalar_labels[index] == 1:
      vector_labels[index,0] = 0
      vector_labels[index,1] = 1
  vector_labels = vector_labels.float()
  return vector_labels

def train_loop(dataloader, model, loss_fn, optimizer, device, cube_size, 
logger, batch_size):
  size = len(dataloader.dataset)
  sum_loss = 0
  batches = 0
  for batch, (X, y) in enumerate(dataloader):
    batches += 1
    # Compute prediction and loss
    X = reshape(X, (X.shape[0],1,cube_size,cube_size,cube_size))
    X = X.float().to(device)
    y = one_hot_vector_labels(y).to(device)
    pred = model(X)
    loss = loss_fn(pred, y)
    sum_loss += loss.item()

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print results after each batch        
    if batch % 1 == 0:
      loss, current = loss.item(), batch * batch_size
      # logger = log(f"Training Batch {batches:>} loss: {loss:>7f}  "
      # f"[{current:>5d}/{size:>5d}]", 
      # logger)
      print(f"Training Batch {batches:>} loss: {loss:>7f}  "
      f"[{current:>5d}/{size:>5d}]")
  return sum_loss/batches, logger

def test_loop(dataloader, model, device, cube_size):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss = 0
  all_predictions = torch.zeros(size).to(device)
  all_targets = torch.zeros(size).to(device)

  with torch.no_grad():
    i = 0
    for X, y in dataloader:
      X = reshape(X, (X.shape[0],1,cube_size,cube_size,cube_size))
      X = X.float().to(device)
      y = one_hot_vector_labels(y).to(device)

      pred = model(X)
      _,predictions = torch.max(pred , 1)
      _,targets = torch.max(y, 1)

      test_loss += loss_fn(pred, y.float()).item()

      j = 0
      while j < len(predictions):
        all_predictions[i+j] = predictions[j]
        all_targets[i+j] = targets[j]
        j += 1
      i += len(predictions)

  test_loss /= num_batches
  return test_loss, all_predictions, all_targets

def validate_loop(dataloader, model, loss_fn, device, cube_size):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  validate_loss= 0
  all_predictions = torch.zeros(size).to(device)
  all_targets = torch.zeros(size).to(device)

  with torch.no_grad():
    i = 0
    for X, y in dataloader:
      X = reshape(X, (X.shape[0],1,cube_size,cube_size,cube_size))
      X = X.float().to(device)
      y = one_hot_vector_labels(y).to(device)
      
      # Find outputs from model for each image in batch X
      pred = model(X)
      print(pred)
      print(y)

      # Convert model predictions to from vector of 2 floats to 1 or 0 and
      # targets from 1 hot vector to 1 or 0
      _,predictions = torch.max(pred , 1)
      _,targets = torch.max(y, 1)

      print(predictions)
      print(targets)

      # Add loss for this batch (divided by number of batches later to return
      # average)
      validate_loss += loss_fn(pred, y.float()).item()
      
      # Loop through predictions and targets for the batch and add to array for 
      # all batches. Brings all results together so they can be returned as 
      # arrays
      j = 0
      while j < len(predictions):
        all_predictions[i+j] = predictions[j]
        all_targets[i+j] = targets[j]
        j += 1
      i += len(predictions)

  validate_loss /= num_batches
  print(all_predictions)
  print(all_targets)
  return validate_loss, all_predictions, all_targets