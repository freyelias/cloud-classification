import random
import csv
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score
from config import get_model_ensemble_info

torch.cuda.init()
torch.autograd.set_detect_anomaly(True)


class CloudDataset(Dataset):
    def __init__(self, img_lab_list, label_category, transform=None):
        self.img_lab_list = img_lab_list
        self.label_category = label_category
        self.transform = transform

    def __len__(self):
        return len(self.img_lab_list)

    def __getitem__(self, index):
        img_path = self.img_lab_list[index][0]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Cloud label classification
        if self.label_category == 'cloud':
            label = torch.tensor(int(self.img_lab_list[index][1]))
        # Cloud base height label classification
        elif self.label_category == 'base':
            label = torch.tensor(int(self.img_lab_list[index][2]))
        else:
            raise ValueError(f'Wrong label category <{self.label_category}>, check config.py')

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label


class EarlyStopping:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_datalists(train_lab_file, valid_lab_file):
    train_data_list_aug = []
    valid_data_list = []
    # Read the CSV file and populate the data list with tuples
    with open(train_lab_file, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) == 3:
                value1 = row[0]
                value2 = int(row[1])
                value3 = int(row[2])
                train_data_list_aug.append((value1, value2, value3))

    with open(valid_lab_file, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) == 3:
                value1 = row[0]
                value2 = int(row[1])
                value3 = int(row[2])
                valid_data_list.append((value1, value2, value3))

    random.shuffle(train_data_list_aug)
    random.shuffle(valid_data_list)
    dataset_list = [f[0] for f in train_data_list_aug] + [f[0] for f in valid_data_list]

    return train_data_list_aug, valid_data_list, dataset_list


def calculate_normalization(image_list):
    mean_values = np.zeros(3)  # Initialize the mean values for each channel
    std_values = np.zeros(3)  # Initialize the standard deviation values for each channel
    num_images = 0

    # Loop through all the images in the dataset
    for path in image_list:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is not None:
            # Convert the image to float32 format for calculations
            img = img.astype(np.float32) / 255.0

            # Calculate the mean and standard deviation for each channel (axis=0)
            mean_values += np.mean(img, axis=(0, 1))
            std_values += np.std(img, axis=(0, 1))

            num_images += 1
        else:
            raise ValueError(f"Warning: Failed to load image '{path}'")

    # Calculate the average mean and standard deviation across all valid images
    if num_images > 0:
        mean_values /= num_images
        std_values /= num_images
    print(f'Normalized mean: {mean_values}, Normalized std: {std_values}')

    return mean_values, std_values


def data_loader(hyperparameters, train_set, valid_set):
    train_loader = DataLoader(dataset=train_set, batch_size=hyperparameters['batch_size'], shuffle=True)
    valid_loader = DataLoader(dataset=valid_set, batch_size=hyperparameters['batch_size'], shuffle=True)

    return train_loader, valid_loader


def get_lr_steps(lr_list):
    if lr_list:
        lr_reductions = {}
        for t in lr_list:
            lr_value = t[1]
            if lr_value not in lr_reductions:
                lr_reductions[lr_value] = t

        # Extract the unique tuples from the dictionary
        lr_reduction_steps = list(lr_reductions.values())
    else:
        lr_reduction_steps = []
    return lr_reduction_steps


def calculate_model_weights(model_output, true_labels, mode='F1'):
    if mode == 'F1':
        # Convert model output to predicted labels
        predicted_labels = torch.argmax(model_output, dim=1).cpu().numpy()

        # Calculate F1 score for each class
        f1_scores = []
        for class_idx in range(model_output.shape[1]):
            f1_scores.append(f1_score(true_labels == class_idx, predicted_labels == class_idx))

        # Normalize F1 scores to sum up to 1 for each model
        total_f1 = sum(f1_scores)
        weights = [score / total_f1 for score in f1_scores]

        return weights
    elif mode == 'accuracy':
        # Convert model output to predicted labels
        predicted_labels = torch.argmax(model_output, dim=1).cpu().numpy()

        # Calculate accuracy for each class
        accuracies = []
        for class_idx in range(model_output.shape[1]):
            # Calculate accuracy for this class
            correct = np.sum((true_labels == class_idx) & (predicted_labels == class_idx))
            total = np.sum(true_labels == class_idx)
            accuracy = correct / total if total != 0 else 0
            accuracies.append(accuracy)

        # Normalize accuracies to sum up to 1 for each model
        total_accuracy = sum(accuracies)
        weights = [accuracy / total_accuracy for accuracy in accuracies]
        return weights
    else:
        raise ValueError(f'Wrong mode input <{mode}> must be <F1> or <accuracy>')


def evaluate_model(model, valid_loader, model_save_name, hyperparameters, device):
    model.load_state_dict(torch.load(f'results/models/{model_save_name}.pth'))
    model.eval()

    # Lists to store predicted and true labels for evaluation
    predicted_labels_list = []
    predicted_probabilities_list = []
    true_labels_list = []

    # Evaluation loop for validation data
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)

            probability, predicted = torch.max(outputs, 1)

            # Append the predicted and true labels to the lists
            predicted_labels_list.extend(predicted.cpu().numpy())
            predicted_probabilities_list.extend(probability.cpu().numpy())
            true_labels_list.extend(targets.cpu().numpy())

    # Convert the lists to numpy arrays
    predicted_labels = np.array(predicted_labels_list)
    true_labels = np.array(true_labels_list)

    # Calculate and print classification report and confusion matrix
    norm_accuracy = accuracy_score(true_labels, predicted_labels, normalize=True)
    class_report = classification_report(true_labels, predicted_labels, output_dict=True)

    mean_precision = sum([class_report[str(r)]['precision'] for r in range(0, hyperparameters['num_classes'])]
                         ) / hyperparameters['num_classes']
    mean_recall = sum([class_report[str(r)]['recall'] for r in range(0, hyperparameters['num_classes'])]
                      ) / hyperparameters['num_classes']
    mean_f1_score = sum([class_report[str(r)]['f1-score'] for r in range(0, hyperparameters['num_classes'])]
                        ) / hyperparameters['num_classes']
    metrics = {
        'Accuracy': round(norm_accuracy, 2),
        'Precision': round(mean_precision, 2),
        'Recall': round(mean_recall, 2),
        'F1-score': round(mean_f1_score, 2),
    }

    return metrics, true_labels, predicted_probabilities_list, predicted_labels, class_report


def save_stats(model, model_type, model_name, valid_loader, train_acc, valid_acc, train_loss, valid_loss, datetime_str,
               runtime, hyperparameters, lr_list, early_stopped, layer_structure, device):
    model_save_name = f'{model_name}_{datetime_str}'
    metrics_list, true_labels, predicted_probabilities_list, predicted_labels, class_report = evaluate_model(
        model, valid_loader, model_save_name, hyperparameters, device)

    if model_type == 'Ensemble':
        _, decision_method = get_model_ensemble_info()
    else:
        decision_method = None

    # Create saving stats dictionary
    stats_data = {
        'model_name': model_name,
        'enseblme_decision_method': decision_method,
        'label_category': hyperparameters['label_category'],
        'training_accuracy': [round(v, 6) for v in train_acc],
        'validation_accuracy': [round(v, 6) for v in valid_acc],
        'training_loss': [round(v, 6) for v in train_loss],
        'validation_loss': [round(v, 6) for v in valid_loss],
        'Epochs': hyperparameters['epochs'],
        'Workers': hyperparameters['num_workers'],
        'Image size': hyperparameters['image_size'],
        'Batch size': hyperparameters['batch_size'],
        'Channels': hyperparameters['in_channels'],
        'Classes': hyperparameters['num_classes'],
        'Optimizer': hyperparameters['optimizer'],
        'Loss function': hyperparameters['criterion'],
        'Scheduler': hyperparameters['scheduler'],
        'Learning rate': hyperparameters['learning_rate'],
        'LR factor': hyperparameters['lr_factor'],
        'LR patience': hyperparameters['lr_patience'],
        'ES patience': hyperparameters['es_patience'],
        'ES mindelta': hyperparameters['es_mindelta'],
        'Accuracy': metrics_list['Accuracy'],
        'Precision': metrics_list['Precision'],
        'Recall': metrics_list['Recall'],
        'F1-score': metrics_list['F1-score'],
        'class_report': class_report,
        'Runtime': runtime,
        'true_labels': true_labels,
        'predicted_labels': predicted_labels,
        'datetime_str': datetime_str,
        'best_valid_acc': max(valid_acc),
        'lr_list': lr_list,
        'early_stopped': early_stopped,
        'model_architecture': str(model),
        'layer_grads': layer_structure,
    }
    file_name = f'results/json/{model_save_name}.json'
    with open(file_name, 'w') as json_file:
        json.dump(stats_data, json_file, cls=NumpyEncoder)
