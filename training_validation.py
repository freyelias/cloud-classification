import numpy as np
from tqdm import tqdm
import torch
import config
import torch.nn.functional as F


def train(model, trainloader, optimizer, criterion, device):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)
        if labels.dtype != torch.long:
            labels = labels.long()
        optimizer.zero_grad()
        # Calculate the loss.
        outputs = model(image)
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()

    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))

    return epoch_loss, epoch_acc


# Validation function.
def validate(model, validloader, criterion, device):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(validloader), total=len(validloader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            if labels.dtype != torch.long:
                labels = labels.long()
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy
            probability, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(validloader.dataset))

    return epoch_loss, epoch_acc


def validate_single_model(model_list, validloader, device):
    model_outputs_list = []
    hyperparameters = config.get_hyperparameters()
    model_accuracy_per_class = []

    for single_model in model_list:
        single_model_outputs = []
        single_model.eval()  # Set the model to evaluation mode
        correct_per_class = [0] * hyperparameters['num_classes']
        total_per_class = [0] * hyperparameters['num_classes']

        with torch.no_grad():
            for inputs, targets in validloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = single_model(inputs)
                single_model_outputs.append(outputs.cpu().numpy())
                _, predicted = torch.max(outputs, 1)

                for t, p in zip(targets, predicted):
                    correct_per_class[t] += (p == t).item()
                    total_per_class[t] += 1
        model_outputs_list.append(single_model_outputs)

        model_accuracy = [correct_per_class[i] / total_per_class[i] for i in range(hyperparameters['num_classes'])]
        model_accuracy_per_class.append(model_accuracy)
    model_outputs_list = np.array(model_outputs_list)
    model_accuracy_per_class = np.array(model_accuracy_per_class)
    return model_outputs_list, model_accuracy_per_class
