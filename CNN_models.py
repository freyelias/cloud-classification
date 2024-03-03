import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
import utils
import training_validation
import dataset
torch.cuda.init()
torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CloudNet(nn.Module):
    def __init__(self, num_classes):
        super(CloudNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.relu1 = nn.ReLU()
        self.banrm1 = nn.BatchNorm2d(96)
        self.mxpl1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.banrm2 = nn.BatchNorm2d(256)
        self.mxpl2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.banrm3 = nn.BatchNorm2d(384)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.banrm4 = nn.BatchNorm2d(256)
        self.mxpl3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.drop1 = nn.Dropout(0.5)
        self.lin1 = nn.Linear(6400, 4096)  # Update the input size here
        self.relu5 = nn.ReLU()
        self.lin2 = nn.Linear(4096, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.banrm1(out)
        out = self.mxpl1(out)
        out = self.relu2(self.conv2(out))
        out = self.banrm2(out)
        out = self.mxpl2(out)
        out = self.relu3(self.conv3(out))
        out = self.banrm3(out)
        out = self.relu4(self.conv4(out))
        out = self.banrm4(out)
        out = self.mxpl3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop1(out)
        out = self.lin1(out)
        out = self.relu5(out)
        out = self.lin2(out)
        return out


class CCNet(nn.Module):
    def __init__(self, num_classes):
        super(CCNet, self).__init__()

        # Convolutional layers for feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully connected layers for cloud type classification
        self.fc_classification = nn.Sequential(
            nn.Linear(401408, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Extract features
        x = self.features(x)
        x = x.view(x.size(0), -1)

        class_scores = self.fc_classification(x)

        return class_scores


class EnsembleModel(nn.Module):
    def __init__(self, models_list, valid_loader, decision_mode='majority_voting', model_weights=None):
        super(EnsembleModel, self).__init__()
        self.models_list = nn.ModuleList(models_list)
        self.valid_loader = valid_loader
        self.num_models = len(models_list)
        self.decision_mode = decision_mode
        self.model_weights = model_weights

    def forward(self, x):
        model_outputs = [model(x) for model in self.models_list]

        # Perform majority voting on the predictions
        if self.decision_mode == 'majority_voting':
            ensemble_predictions = torch.stack(model_outputs, dim=2)
            majority_votes, _ = ensemble_predictions.mode(dim=2)

            # Compute prediction probabilities using softmax
            # prediction_probs = F.softmax(ensemble_predictions, dim=1)
            print(majority_votes)
            return majority_votes.squeeze(dim=2)

        elif self.decision_mode == 'averaging':
            ensemble_predictions = torch.stack(model_outputs, dim=2)
            averaged_predictions = torch.mean(ensemble_predictions, dim=2)
            return averaged_predictions

        elif self.decision_mode == 'weighting':
            if self.model_weights is None:
                raise ValueError("No weights are provided for model decision mode <weighting>")

            if len(self.model_weights) != self.num_models:
                raise ValueError(f"Number of weights <{len(self.model_weights)}> "
                                 f"should match number of models <{self.num_models}>")

            weighted_sum = torch.zeros_like(model_outputs[0])
            for i in range(self.num_models):
                weighted_sum += model_outputs[i] * self.model_weights[i]
            print(weighted_sum)
            return weighted_sum

        elif self.decision_mode == 'probability_weighting':
            # Get model outputs
            model_outputs = [model(x) for model in self.models_list]

            # Initialize empty tensor to store weighted predictions
            weighted_predictions = torch.zeros_like(model_outputs[0]).to(x.device)

            # Apply weights based on predicted class for each model
            for i in range(self.num_models):
                # Get weights of current model for each class
                weights = self.model_weights[i]
                #
                for j in range(len(weights)):
                    weighted_predictions[:, j] += weights[j] * model_outputs[i][:, j]

            # Normalize the predictions by sum of weights for each class
            # Calculate sum of weights for each class
            sum_weights_per_class = torch.tensor(self.model_weights).sum(dim=0).to(x.device)
            # Ensure non-zero weights to avoid division by zero
            sum_weights_per_class = torch.where(sum_weights_per_class == 0, torch.tensor(1.0).to(x.device),
                                                sum_weights_per_class)
            ensemble_prediction = weighted_predictions / sum_weights_per_class

            return ensemble_prediction

        elif self.decision_mode == 'class_weighting':
            # Get model outputs
            model_outputs = [model(x) for model in self.models_list]

            # Initialize empty tensor to store weighted predictions
            weighted_predictions = torch.zeros_like(model_outputs[0]).to(x.device)

            # Apply weights based on predicted class for each model
            for i in range(self.num_models):
                # Get weights of current model for each class
                weights = self.model_weights[i]
                #
                for j in range(len(weights)):
                    weighted_predictions[:, j] += weights[j] * model_outputs[i][:, j]

            # Normalize the predictions by sum of weights for each class
            # Calculate sum of weights for each class
            sum_weights_per_class = torch.tensor(self.model_weights).sum(dim=0).to(x.device)
            # Ensure non-zero weights to avoid division by zero
            sum_weights_per_class = torch.where(sum_weights_per_class == 0, torch.tensor(1.0).to(x.device),
                                                sum_weights_per_class)
            ensemble_prediction = weighted_predictions / sum_weights_per_class

            return ensemble_prediction


def load_model(model_type, model_version, hyperparameters):
    if model_type == 'AlexNet':
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        model_name = f'AlexNet (pretrained)'

    elif model_type == 'CCNet':
        model = CCNet(hyperparameters['num_classes'])
        model_name = 'CCNet'

    elif model_type == 'CloudNet':
        model = CloudNet(hyperparameters['num_classes'])
        model_name = f'CloudNet'

    elif model_type == 'ConvNeXt':
        model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT)
        model_name = f'ConvNeXt (pretrained)'

    elif model_type == 'DenseNet':
        model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
        model_name = f'DenseNet{model_version} (pretrained)'

    elif model_type == 'MaxVit':
        model = models.maxvit_t(weights=models.MaxVit_T_Weights.DEFAULT)
        model_name = f'MaxViT (pretrained)'

    elif model_type == "ResNet":
        if model_version == 18:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif model_version == 34:
            model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif model_version == 50:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif model_version == 101:
            model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        elif model_version == 152:
            model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        else:
            raise ValueError("Invalid ResNet version, should be one of [18, 34, 50, 101, 152]")
        model_name = f'ResNet{model_version} (pretrained)'

    elif model_type == 'ResNeXt':
        if model_version == 50:
            model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
            model_name = f'ResNeXt{model_version} (pretrained)'
        else:
            raise ValueError("Invalid ResNeXt version, should be '50'")

    elif model_type == 'ShuffleNet':
        model = models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.DEFAULT)
        model_name = f'ShuffleNetV2 (pretrained)'

    elif model_type == 'VGG':
        if model_version == '19bn':
            model = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT)
            model_name = f'VGG{model_version} (pretrained)'
        else:
            raise ValueError("Invalid VGG version, should be '19bn'")
    else:
        raise ValueError(
            "Invalid <model_type>, check config.py (should be of format e.g. <ResNet>)")

    return model, model_name
