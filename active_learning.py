# region Initialization
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import shutil
import platform
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from matplotlib.patches import Circle
import pickle
import hashlib
import json

# Custom imports
from utils.preprocessing import DatasetHash, custom_collate
from utils.architectures import EffNetB0, CustomUnet
from utils.DCoM import DCoM

# Set seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# mode = 'segmentation'
# data_path = r'C:\Users\Alexandre Bonin\Documents\Stage\datasets\Dataset_fixe_2024'
# model_path = r'C:\Users\Alexandre Bonin\Documents\Stage\Code segmentation (Leo)\Models\VL_2024+basemodelproxicam.pt'
# model = CustomUnet(encoder_name="resnet50",
#             encoder_weights="imagenet",
#             classes = 2,
#             activation=None,
#             encoder_depth=5,
#             decoder_channels=[256, 128, 64, 32, 16],
#             extract_features=True
#             ).to(device)

mode = 'classification'
data_path = r'/home/abonin/Desktop/datasets/sample_Luchey'
model_path = r'/home/abonin/Desktop/Classification-model/models/Run_2024-11-08_16-18-18.pth'
model = EffNetB0(num_classes=2, model_path = model_path, extract_features = True).to(device)
model.eval()


budgetSize = 1000
batch_size = 128

output_dir = os.path.join(data_path, 'DCoM')
metadata_path = os.path.join(output_dir, 'metadata.json')
samples_dir = os.path.join(output_dir, 'samples')

if not os.path.exists(data_path):
    raise FileNotFoundError(f'Data path not found: {data_path}')
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir, exist_ok=True)

model.eval()


def compute_image_hash(image_path):
    with open(image_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()
    
def update_metadata(data_path, metadata_path):

    print('\n','Updating metadata...')
    # Initialize metadata
    metadata = {}
    added = 0
    removed = 0
    updated = 0
    
    output_dir = os.path.join(data_path + '/DCoM')
    samples_dir = os.path.join(output_dir, 'samples')

    # Read metadata if it exists
    if os.path.exists(metadata_path):
        try :
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            print('Error reading metadata, starting from scratch.')
    
    #Update metadata with unlabeled images
    for image in os.listdir(data_path) :
        if image.endswith(('.jpg', '.jpeg')):
            img_path = os.path.join(data_path, image)
            img_hash = compute_image_hash(img_path)
        
            if img_hash not in metadata:
                metadata[img_hash] = {'state': 'unlabeled', 'path': img_path}
                added += 1

    # Update metadata with labeled images
    for root, _, files in os.walk(samples_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                img_hash = compute_image_hash(img_path)
                if img_hash not in metadata:
                    metadata[img_hash] = {'state': 'labeled', 'path': img_path}
                    added += 1
                elif metadata[img_hash]['state'] != 'labeled':
                    metadata[img_hash]['state'] = 'labeled'
                    updated += 1             

    # Validate that all images exist and update metadata if not
    for img_hash, info in list(metadata.items()):  # Use list() to avoid runtime modification
        if not os.path.exists(info['path']):
            del metadata[img_hash]
            removed += 1
   
    # Save updated metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f'Added {added} entries, removed {removed} entries, updated {updated} entries. \n')

    return metadata

#endregion #################################################################################################################
#region Feature extraction

if __name__ == '__main__':
   
    metadata = update_metadata(data_path, metadata_path)

    features = {}
    pseudo_labels = {}
    confidences = {}

    # Define unlabeled set and labeled set
    uSet = [h for h, info in metadata.items() if info['state'] != 'labeled']
    lSet = [h for h, info in metadata.items() if info['state'] == 'labeled']

    if platform.system() == "Linux" and mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

    num_workers = (6 if device.type == 'cuda' else 0)
    print(f'Number of workers: {num_workers}')

    # Define the transform and dataset
    dataset = DatasetHash(metadata, device=device)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=custom_collate)
    print(f'Number of images: {len(dataset)} \n')
    if len(lSet) == len(dataset):
        print('All images are labeled, nothing to do.')
        exit()

    print('Starting feature extraction...', end='\n')

    for h_batch, img_batch in tqdm(data_loader):
        # Filter out None values caused by skipped corrupted images
        hashes_batch = [h for h, img in zip(h_batch, img_batch) if img is not None]
        img_batch = [img for img in img_batch if img is not None]

        if not img_batch:  # Skip empty batches
            continue

        img_batch = torch.stack(img_batch).to(device)  # Batch all valid images together on GPU

        with torch.no_grad():
            # Get model predictions and features
            pred_batch, feature_batch = model(img_batch)
            pred_batch = torch.softmax(pred_batch, dim=1)

        for img_hash, pred, feature in zip(hashes_batch, pred_batch, feature_batch):
            if mode == 'segmentation':
                # Compute max probability and positive proportion for segmentation
                max_probs, binary = torch.max(pred, dim=0)
                avg_confidence = max_probs.mean().item()
                positive_proportion = (binary == 1).float().mean().item()
                pseudo_labels[img_hash] = positive_proportion
                confidences[img_hash] = avg_confidence
            elif mode == 'classification':
                # Classification case: get the predicted class label
                predicted_class = pred.argmax(dim=0).item()
                pseudo_labels[img_hash] = predicted_class
                confidences[img_hash] = pred.max().item()
            
            features[img_hash] = feature.cpu().numpy()

    # Ensure features and pseudo_labels have the same number of rows
    assert len(features) == len(pseudo_labels), f"Mismatch: features has {len(features)} rows, pseudo_labels has {len(pseudo_labels)} rows"

    if mode == 'segmentation':
            bins = [-np.inf, 0.1, 0.3, 0.5, 0.8, np.inf]
            cut_labels = pd.cut(list(pseudo_labels.values()), bins, labels=[0, 1, 2, 3, 4]).astype(int)
            pseudo_labels = dict(zip(pseudo_labels.keys(), cut_labels))

    # Convert the features to a numpy array
    features_transform = np.array(list(features.values()))

    # Flatten each feature
    N = features_transform.shape[0]
    features_transform = features_transform.reshape(N, -1)
    # Standardize the features
    scaler = Normalizer()
    features_transform = scaler.fit_transform(features_transform)
    # Use randomized solver in PCA for acceleration
    if features_transform.shape[1] > 10e5:
        print(f'Old features.shape : {features_transform.shape}')
        pca = PCA(n_components=min(features_transform.shape[0], 1000), svd_solver='randomized')
        features_transform = pca.fit_transform(features_transform)
        print(f'New features.shape : {features_transform.shape}')

    #Map features back to hash values
    features = dict(zip(list(features.keys()), features_transform))

    #endregion #################################################################################################################
    #region Dcom sample images
    lSet_deltas = {h : metadata[h]['delta'] for h in lSet}

    dcom = DCoM(features, lSet, budgetSize = min(budgetSize, len(dataset) - len(lSet)), lSet_deltas=lSet_deltas)
    active_set, new_uset, lSet_deltas = dcom.select_samples(confidences)

    # Update the metadata with the new active set
    for h in active_set:
        metadata[h]['delta'] = lSet_deltas[h]
        metadata[h]['state'] = 'active'
    
    # Save updated metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    # Save the sampled images to a new directory
    for h in active_set:
        img_path = metadata[h]['path']

        if mode == 'classification':
            img_label = pseudo_labels[h]
            img_label_dir = os.path.join(samples_dir, str(img_label))
            os.makedirs(img_label_dir, exist_ok=True)
            shutil.copy(img_path, os.path.join(img_label_dir, os.path.basename(img_path)))
    
        elif mode == 'segmentation':
            shutil.copy(img_path, os.path.join(samples_dir, os.path.basename(img_path)))

    print('Breakpoint : label the images, then type c and press Enter in the terminal to continue.')
    breakpoint()
    
    #endregion #################################################################################################################
    #region Dcom update Deltas

    metadata = update_metadata(data_path, metadata_path)

    features = {h : features[h] for h in metadata if h in features}
    pseudo_labels = {h : pseudo_labels[h] for h in metadata if h in pseudo_labels}
    img_list = {h : metadata[h]['path'] for h in metadata}

    # Define unlabeled set and labeled set
    uSet = [h for h, info in metadata.items() if info['state'] == 'unlabeled']
    lSet = [h for h, info in metadata.items() if info['state'] == 'labeled']

    sample_lSet = random.sample(lSet, min(10, len(lSet)))

    print(f'Size of lSet : {len(lSet)}')
    print(f'lSet_labels : {[pseudo_labels[h] for h in sample_lSet]}')
    print(f'lSet_deltas : {[lSet_deltas[h] for h in sample_lSet]}')

    dcom = DCoM(features, lSet, budgetSize = budgetSize, lSet_deltas=lSet_deltas)
    lSet_deltas = dcom.new_centroids_deltas(pseudo_labels)

    for img_hash, delta in lSet_deltas.items():
        metadata[img_hash]['delta'] = delta
    
        # Convert float32 values in metadata to standard float
    for img_hash, info in metadata.items():
        if 'delta' in info and isinstance(info['delta'], np.float32):
            info['delta'] = float(info['delta'])

    # Save updated metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print('Finished udating deltas \n')

    #endregion #################################################################################################################
    #region Plot features

    print('Plotting features...')
    
    # Create a consistent list of hashes
    hashes = list(features.keys())

    # Build the feature matrix in the same order as 'hashes'
    feature_matrix = np.array([features[h] for h in hashes])
    # Perform t-SNE on the feature matrix
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(feature_matrix)

    # Create a mapping from hash to index
    hash_to_idx = {h: idx for idx, h in enumerate(hashes)}
    # Extract indices for 'active_set' and 'lSet' using the mapping
    active_indices = [hash_to_idx[h] for h in active_set if h in hash_to_idx]
    lSet_indices = [hash_to_idx[h] for h in lSet if h in hash_to_idx]

    # Extract the 2D coordinates for the sampled and labeled points
    sampled_points = features_2d[active_indices]
    labeled_points = features_2d[lSet_indices]
    
    S = min(15000 / features_2d.shape[0], 30)

    plt.figure(figsize=(12, 6))

    # Add circles of radius delta around points from lSet
    ax = plt.gca()
    for point_idx in lSet_indices:
        point_2d = features_2d[point_idx]
        delta = float(lSet_deltas[hashes[point_idx]])
        circle = Circle(point_2d, radius=delta, fill=True, facecolor='lightblue', alpha=0.8, linestyle='-', linewidth=0.6)
        ax.add_patch(circle)

    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=list(pseudo_labels.values()), cmap='viridis', s=S, label = 'Predicted pseudo-labels')

    # Highlight the sampled points and labeled points in a different color or marker (e.g., red stars)
    plt.scatter(labeled_points[:, 0], labeled_points[:, 1], c='green', edgecolors='green', marker='*', s=S*1.04, label='Labeled points')
    plt.scatter(sampled_points[:, 0], sampled_points[:, 1], c='red', edgecolors='red', marker='*', s=S*1.05, label='Sampled points')

    plt.colorbar()
    plt.title("Features Colored by Predicted Labels")
    plt.legend()

    # Save the plot to the dcom folder
    plot_path = os.path.join(output_dir, 'features_plot.png')
    plt.savefig(plot_path)
    plt.show()




