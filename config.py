import os
import torch
from torch import nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ______________________________________________________________________________________________________________________
# Define parameters

# Model type and version
# Define model type [AlexNet, CCNet, CloudNet, ConvNeXt, DenseNet, MaxVit, ResNet, ResNeXt, ShuffleNet, VGG, Ensemble]
# And model version [   ''     ''       ''     'large'        169      ''    18-152     50         ''     19bn     ''  ]
model_type = 'Ensemble'
decision_method = 'class_weighting'       # ['majority_voting', 'averaging', 'probability_weighting', 'class_weighting']
model_version = 50

# Dictionary of models if Ensemble Training is specified (model_type = 'Ensemble')
ensemble_models = {
    'ConvNeXt': ('', [0.93, 0.87, 0.87, 0.87, 0.67, 0.53, 0.93, 0.53, 0.93, 0.53, 1]),
    'MaxVit': ('', [0.93, 0.8, 0.53, 0.87, 0.4, 0.73, 0.93, 0.6, 0.73, 0.6, 0.93]),
    'ResNet': (50, [0.93, 0.87, 0.87, 1, 0.6, 0.4, 0.93, 0.55, 0.67, 0.53, 1]),
    'ShuffleNet': ('', [0.87, 1, 0.93, 0.87, 0.67, 0.4, 0.73, 0.4, 0.87, 0.6, 1]),
    'VGG': ('19bn', [1, 0.87, 0.93, 1, 0.53, 0.6, 0.93, 0.53, 0.8, 0.73, 1]),
}

# Model parameters
label_category = 'cloud'                    # Define classification task ['cloud', 'base']

# Files
training_labels = \
    r'data/UBELIX_training_images_aug.csv'    # Path to training labels (csv-file)
validation_labels = \
    r'data/UBELIX_validation_images.csv'      # Path to validation labels (csv-file)

# Hyperparameters
epochs = 4                                  # Number of training epochs
num_workers = 4                             # Number of workers
image_size = 224                            # Image size
batch_size = 32                             # Batch size
in_channels = 3                             # Number of channels (3 for RGB)
optimizer_type = 'SGD'                      # Backpropagation method
weight_decay = 0.01                         # Weight decay (L1/L2 regularization) for Optimizer
learning_rate = 0.001                       # Learning rate
scheduler_type = 'ReduceLROnPlateau'        # Learning Scheduler [StepLR or ReduceLROnPlateau]
lr_factor = 0.1                             # Learning rate reduction factor
lr_patience = 25                            # Learning rate step in epochs
early_stopping = True                       # Early Stopping
es_patience = 100                           # Early Stopping tolerance in epochs
es_mindelta = 0.0001                        # Early Stopping non-improvement tolerance
# ______________________________________________________________________________________________________________________


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_paths():
    if not os.path.exists("results/plots"):
        os.makedirs("results/plots")
    if not os.path.exists("results/json"):
        os.makedirs("results/json")
    if not os.path.exists("results/models"):
        os.makedirs("results/models")
    train_lab_file = training_labels
    valid_lab_file = validation_labels

    return train_lab_file, valid_lab_file


def get_transforms(norm_mean, norm_std, img_size):
    if img_size != 360:
        train_transforms = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=norm_mean, std=norm_std),  # Normalize the image
            ToTensorV2(),  # Convert the image to a PyTorch tensor
        ])
        valid_transforms = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=norm_mean, std=norm_std),  # Normalize the image
            ToTensorV2(),  # Convert the image to a PyTorch tensor
        ])

    else:
        train_transforms = A.Compose([
            A.Normalize(mean=norm_mean, std=norm_std),  # Normalize the image
            ToTensorV2(),  # Convert the image to a PyTorch tensor
        ])

        valid_transforms = A.Compose([
            A.Normalize(mean=norm_mean, std=norm_std),  # Normalize the image
            ToTensorV2(),  # Convert the image to a PyTorch tensor
        ])
    print(f'Transformation image size is {img_size} for model type {model_type}')

    return train_transforms, valid_transforms


def get_model_params(model, hyperparameters):
    # Optimizer
    if hyperparameters['optimizer_type'] == 'SGD':
        optimizer = SGD(model.parameters(), lr=hyperparameters['learning_rate'],
                        weight_decay=hyperparameters['weight_decay'])
    elif hyperparameters['optimizer_type'] == 'Adam':
        optimizer = Adam(model.parameters(), lr=hyperparameters['learning_rate'], betas=(0.9, 0.999), eps=1e-08,
                         weight_decay=hyperparameters['weight_decay'], amsgrad=False)
    elif hyperparameters['optimizer_type'] == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=hyperparameters['learning_rate'], betas=(0.9, 0.999), eps=1e-08,
                          weight_decay=hyperparameters['weight_decay'], amsgrad=False)
    else:
        raise ValueError(f'Optimizer not found, should be one of [SGD, Adam]')

    # Loss function.
    criterion = nn.CrossEntropyLoss()

    # Initialize the scheduler variable
    # Set up the chosen scheduler
    if hyperparameters['optimizer_type'] != 'Adam' or 'AdamW':
        if hyperparameters['scheduler_type'] == 'StepLR':
            scheduler = StepLR(optimizer=optimizer, step_size=hyperparameters['lr_patience'],
                               gamma=hyperparameters['lr_factor'], verbose=False)
        elif hyperparameters['scheduler_type'] == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=hyperparameters['lr_factor'],
                                          patience=hyperparameters['lr_patience'], verbose=True)
        else:
            raise ValueError(f'Scheduler not found, should be one of [StepLR, ReduceLROnPlateau]')
    else:
        scheduler = None

    return optimizer, criterion, scheduler


def freeze_layers(model, num_freezing_layers):
    for i, param in enumerate(model.parameters()):
        if i < num_freezing_layers:
            param.requires_grad = False


def get_model_structure(model, sel_model_type, hyperparameters):
    if sel_model_type == 'AlexNet':
        # Define the number of layers to freeze from the beginning
        num_freezing_layers = 31

        # Freeze layers up to the specified number
        freeze_layers(model, num_freezing_layers)
        # Modify the last fully connected (fc) layer for your number of classes
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, hyperparameters['num_classes'])

    if sel_model_type == 'ConvNeXt':
        # Define the number of layers to freeze from the beginning
        num_freezing_layers = 31

        # Freeze layers
        freeze_layers(model, num_freezing_layers)
        # Modify the last fully connected (fc) layer for your number of classes
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, hyperparameters['num_classes'])

    if sel_model_type == 'DenseNet':
        # Freeze layers
        num_freezing_layers = 42

        # Freeze layers
        freeze_layers(model, num_freezing_layers)
        # Modify the last fully connected (fc) layer for your number of classes
        model.classifier = nn.Linear(model.classifier.in_features, hyperparameters['num_classes'])
        # Enable the gradient computation for the new fully connected layer
        for param in model.classifier.parameters():
            param.requires_grad = True

    if sel_model_type == 'MaxVit':
        num_freezing_layers = 8
        # Freeze layers
        freeze_layers(model, num_freezing_layers)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, hyperparameters['num_classes'])
        for param in model.classifier.parameters():
            param.requires_grad = True

    if sel_model_type == 'ResNet':
        # Define the number of layers to freeze from the beginning
        num_freezing_layers = 24

        # Freeze layers
        freeze_layers(model, num_freezing_layers)

        # Added FC layer(s)
        model.fc = nn.Linear(model.fc.in_features, hyperparameters['num_classes'])

        # Enable the gradient computation for the new fully connected layer
        for param in model.fc.parameters():
            param.requires_grad = True

    if sel_model_type == 'ResNeXt':
        # Freeze layers
        num_freezing_layers = 8

        # Freeze layers
        freeze_layers(model, num_freezing_layers)

        # Modify the last fully connected (fc) layer for your number of classes
        model.fc = nn.Linear(model.fc.in_features, hyperparameters['num_classes'])

    if sel_model_type == 'ShuffleNet':
        num_freezing_layers = 18  # 18 / 45

        freeze_layers(model, num_freezing_layers)
        model.fc = nn.Linear(model.fc.in_features, hyperparameters['num_classes'])
        for param in model.fc.parameters():
            param.requires_grad = True

    if sel_model_type == 'VGG':
        # Freeze the first few layers
        num_freezing_layers = 9

        freeze_layers(model, num_freezing_layers)
        model.classifier[-1] = nn.Linear(4096, hyperparameters['num_classes'])
        for param in model.classifier[-1].parameters():
            param.requires_grad = True

    # Save layer grads
    layer_grads = {}
    for name, param in model.named_parameters():
        layer_grads[name] = {"requires_grad": param.requires_grad}

    return model, layer_grads


def get_hyperparameters():
    hyperparameters = {
        'label_category': label_category,
        'epochs': epochs,
        'num_workers': num_workers,
        'image_size': 224,
        'batch_size': batch_size,
        'in_channels': in_channels,
        'num_classes': 11 if label_category == 'cloud' else (
            4 if label_category == 'base' else ValueError(f"Invalid label_category '{label_category}'")),
        'optimizer_type': optimizer_type,
        'weight_decay': weight_decay,
        'learning_rate': learning_rate,
        'scheduler_type': scheduler_type if 'Adam' not in optimizer_type else 'None',
        'lr_factor': lr_factor,
        'lr_patience': lr_patience,
        'early_stopping': early_stopping,
        'es_patience': es_patience,
        'es_mindelta': es_mindelta,
    }

    return hyperparameters


def update_hyperparameters(hyperparameters, optimizer, criterion, scheduler):
    update_parameters = {
        'optimizer': str(type(optimizer).__name__),
        'criterion': str(type(criterion).__name__),
        'scheduler': str(type(scheduler).__name__) if 'Adam' not in hyperparameters['optimizer_type'] else 'None',
    }
    hyperparameters.update(update_parameters)
    return hyperparameters


def get_legend_hyperparameters(hyperparameters):
    hyperparameters_legend = {
        # 'Label category': hyperparameters['label_category'],
        'Epochs': hyperparameters['epochs'],
        'Image size': hyperparameters['image_size'],
        'Batch size': hyperparameters['batch_size'],
        'Channels': hyperparameters['in_channels'],
        'Classes': hyperparameters['num_classes'],
        'Optimizer': hyperparameters['optimizer'],
        'Weight decay': format(hyperparameters['weight_decay'], '.2g'),
        'Loss function': hyperparameters['criterion'],
        'Scheduler': hyperparameters['scheduler'] if 'Adam' not in hyperparameters['optimizer_type'] else '-',
        'Learning rate': format(hyperparameters['learning_rate'], '.2g'),
        'LR factor': hyperparameters['lr_factor'] if 'Adam' not in hyperparameters['optimizer_type'] else '-',
        'LR patience': hyperparameters['lr_patience'] if 'Adam' not in hyperparameters['optimizer_type'] else '-',
        'ES patience': hyperparameters['es_patience'] if hyperparameters['early_stopping'] else '-',
        'ES mindelta': format(hyperparameters['es_mindelta'], '.2g') if hyperparameters['early_stopping'] else '-',
    }
    return hyperparameters_legend


def get_modelinfo():
    return model_type, model_version


def get_model_ensemble_info():
    return ensemble_models, decision_method


def get_labelcategories():
    cloud_info = {
        0: {'abbr': 'Ci', 'label': 'Cirrus', 'height': 6000, 'base_cat': 3},
        1: {'abbr': 'Cc', 'label': 'Cirrocumulus', 'height': 6000, 'base_cat': 3},
        2: {'abbr': 'Cs', 'label': 'Cirrostratus', 'height': 6000, 'base_cat': 3},
        3: {'abbr': 'Ac', 'label': 'Altocumulus', 'height': 2500, 'base_cat': 2},
        4: {'abbr': 'As', 'label': 'Altostratus', 'height': 2500, 'base_cat': 2},
        5: {'abbr': 'Ns', 'label': 'Nimbostratus', 'height': 2500, 'base_cat': 2},
        6: {'abbr': 'Sc', 'label': 'Stratocumulus', 'height': 0, 'base_cat': 1},
        7: {'abbr': 'St', 'label': 'Stratus', 'height': 0, 'base_cat': 1},
        8: {'abbr': 'Cu', 'label': 'Cumulus', 'height': 0, 'base_cat': 1},
        9: {'abbr': 'Cb', 'label': 'Cumulonimbus', 'height': 0, 'base_cat': 1},
        10: {'abbr': 'Cf', 'label': 'Cloudfree', 'height': 9999, 'base_cat': 0}
    }
    base_name = {
        0: {'abbr': 'CF', 'label': 'Cloudfree', 'height': 9999, 'cloudtypes': ['Cf']},
        1: {'abbr': 'LB', 'label': 'Lower base', 'height': 0, 'cloudtypes': ['Sc', 'St', 'Cu', 'Cb']},
        2: {'abbr': 'MB', 'label': 'Middle base', 'height': 2500, 'cloudtypes': ['Ac', 'As', 'Ns']},
        3: {'abbr': 'HB', 'label': 'Higher base', 'height': 6000, 'cloudtypes': ['Ci', 'Cc', 'Cs']},
    }
    return cloud_info, base_name
