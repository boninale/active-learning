'''
This file contains custom models architectures. 
It is meant to be imported in the training script using : 

from architectures import EffNetB0
model = EffNetB0(num_classes=3, model_path)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import  models
import segmentation_models_pytorch as smp



# EfficientNetB0 Model with custom head
class EffNetB0(nn.Module):
    def __init__(self, num_classes=3, model_path=None, extract_features=False):
        super(EffNetB0, self).__init__()
        self.model_path = model_path
        self.num_classes = num_classes
        self.features = extract_features
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load with pretrained weights from ImageNet
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Modify the classifier with custom layers
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.SiLU(),
            nn.LayerNorm(512),
            nn.Dropout(p=0.4),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Dropout(p=0.3),
            nn.Linear(256, self.num_classes),
        )
        
        if self.model_path:
            try:
                # Load the model after changing the head
                self.load_state_dict(torch.load(self.model_path, map_location=device))
            except RuntimeError as e:
                print(f"Error loading state dictionary from {self.model_path}. Loading the weights with strict=False. You can change the keys' prefix with changestatedict.py. : {e}")
                self.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
            except AttributeError as e:
                print(f"Error loading state dictionary from {self.model_path}: {e}. Ensure the file path is correct and the file is seekable.")

    def forward(self, x):
        if self.features:
            return self.model(x), self.model.features(x)
        else:
            return self.model(x)
    
class TLregularizer(nn.Module):

    def __init__(self, num_classes=3, model_path=None, return_features = True):
        super(TLregularizer, self).__init__()
        self.model_path = model_path
        self.num_classes = num_classes
        self.embedding_dim = num_classes
        self.return_features = return_features
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Remove the original classifier
        self.model.classifier = nn.Identity()

        # Head A: Classification head
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Pool to (1x1x1280)
        self.fc_classification = nn.Sequential(
            nn.Linear(1280, 512),
            nn.SiLU(),
            nn.LayerNorm(512),
            nn.Dropout(p=0.4),
            nn.Linear(512, self.num_classes)  # Final classification logits
        )

        # Head B: Embedding head
        self.fc_embedding = nn.Sequential(
            nn.Flatten(),  # Flatten (7x7x1280 -> 62720)
            nn.Linear(62720, 1024),  # Intermediate layer
            nn.SiLU(),
            nn.LayerNorm(1024),
            nn.Dropout(p=0.4),
            nn.Linear(1024, num_classes)  # Embedding dimension (e.g., 3)
        )

        # Combine classification and embedding outputs
        self.fc_combined = nn.Sequential(
            nn.Linear(num_classes + num_classes, 512),  # Combine logits + embeddings
            nn.SiLU(),
            nn.LayerNorm(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)  # Final prediction
        )

        if self.model_path:
            try:
                # Load the model after changing the head
                self.load_state_dict(torch.load(self.model_path, map_location=device))
            except RuntimeError as e:
                print(f"Error loading state dictionary from {self.model_path}. Loading the weights with strict=False. You can change the keys' prefix with changestatedict.py. : {e}")
                self.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
            except AttributeError as e:
                print(f"Error loading state dictionary from {self.model_path}: {e}. Ensure the file path is correct and the file is seekable.")

    def forward(self, x):
        # Shared feature extraction
        features = self.model.features(x)

        # Head A: Classification logits
        x_a = self.avg_pool(features)
        x_a = torch.flatten(x_a, 1)
        logits = self.fc_classification(x_a)

        # Head B: Embedding
        x_b = self.fc_embedding(features)
        x_b = F.normalize(x_b, p=2, dim=1)  # Normalize embeddings

        # Combine logits and embeddings
        combined = torch.cat([logits, x_b], dim=1)  # Concatenate along feature dimension

        # Final prediction
        final_prediction = self.fc_combined(combined)

        if self.return_features:
            return final_prediction, x_a, x_b
        else:
            return final_prediction
    
# Reference for the model structure (re-implemented in pytorch):
# Taha, Ahmed and Chen, Yi-Ting and Misu, Teruhisa and Shrivastava, Abhinav and Davis, Larry.
# "Boosting Standard Classification Architectures Through a Ranking Regularizer."
# The IEEE Winter Conference on Applications of Computer Vision, 2020, pp. 758-766.

class CustomUnet(nn.Module):
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet", classes=1, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16], extract_features=False):
        super(CustomUnet, self).__init__()
        self.extract_features = extract_features
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            activation=activation,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
        )

    def forward(self, x):
        features = self.model.encoder(x)
        decoder_output = self.model.decoder(*features)
        masks = self.model.segmentation_head(decoder_output)
        
        if self.extract_features:
            # Ensure features are concatenated correctly if they are a list of tensors
            if isinstance(features, list):
                features = torch.cat([f.reshape(f.size(0), -1) for f in features], dim=1)
            return masks, features
        else:
            return masks