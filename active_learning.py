# region imports
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import shutil
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from matplotlib.patches import Circle
import pickle

# Custom imports
from utils.preprocessing import ImageDataset, custom_collate
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
data_path = r'C:\Users\Alexandre Bonin\Documents\Stage\datasets\ProspectFD\sample_PFD'
model_path = r'C:\Users\Alexandre Bonin\Documents\Stage\Classification-model-IMS\models\Run_2024-10-16_14-15-17.pth'
model = EffNetB0(num_classes=2, model_path = model_path, extract_features = True).to(device)

output_dir = os.path.join(data_path + '/DCoM')


if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

model.eval()

#endregion

if __name__ == '__main__':

    #region Feature extraction
    features = []
    pseudo_labels = []

    features_file = os.path.join(output_dir, 'features.pkl')
    pred_file = os.path.join(output_dir, 'pred_df.pkl')
    pseudo_labels_file = os.path.join(output_dir, 'pseudo_labels.pkl')

    # Iterate over the images in the directory
    img_list = [fname for fname in os.listdir(data_path) if fname.endswith(('jpg', 'jpeg', 'png'))]
    img_list = [img_name for img_name in img_list[:500]]

    # # Check if the pickle files exist
    # if os.path.exists(features_file) :
    #     with open(features_file, 'rb') as f:
    #         features = pickle.load(f)

    # if os.path.exists(pseudo_labels_file):
    #     with open(pseudo_labels_file, 'rb') as f:
    #         pseudo_labels = pickle.load(f)

    # if os.path.exists(pred_file): #Make sure it is the right file
    #     print("Loading features and predictions from pickle files...")
    #     # Load the variables from the pickle files
    #     with open(pred_file, 'rb') as f:
    #         pred_df = pickle.load(f)
        
    # else :
    # Initialize a DataFrame to store file paths and prediction probabilities
    pred_df = pd.DataFrame(columns=['img_name', 'pred', 'delta', 'true_label'])
    # pred_df['img_name'] = img_list

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_workers = (4 if device.type == 'cuda' else 0)

    # Define the transform and dataset
    dataset = ImageDataset(img_list, data_path, device=device)
    print(f'Number of images: {len(dataset)}')

    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=custom_collate)

    # Process images in batches
    model = model.to(device)
    model.eval()

    # Batch processing to optimize GPU usage
    for batch in tqdm(data_loader):

        if batch is None:
            continue  # Skip batches with no valid samples

        img_names, img_batch = batch
        img_batch = img_batch.to(device)  # Move batch to GPU
        

        with torch.no_grad():
            # Get model predictions and features
            pred_batch, feature_batch = model(img_batch)
            pred_batch = torch.softmax(pred_batch, dim=1)

        new_rows = []

        for i, img_name in enumerate(img_names):

            feature = feature_batch[i].squeeze().cpu().numpy()
            pred = pred_batch[i]

            # Append features and handle segmentation/classification cases
            features.append(feature)

            if mode == 'segmentation':
                # Compute max probability and positive proportion for segmentation
                max_probs, binary = torch.max(pred, dim=0)
                avg_confidence = max_probs.mean().item()
                positive_proportion = (binary == 1).float().mean().item()
                pseudo_labels.append(positive_proportion)
                # pred_df.loc[pred_df['img_name'] == img_name, 'pred'] = avg_confidence
                new_rows.append({'img_name': img_name, 'pred': avg_confidence, 'delta': None, 'true_label': None})

            elif mode == 'classification':
                # Classification case: get the predicted class label
                predicted_class = pred.argmax(dim=0).item()
                pseudo_labels.append(predicted_class)
                # pred_df.loc[pred_df['img_name'] == img_name, 'pred'] = pred.max().cpu().item()
                new_rows.append({'img_name': img_name, 'pred': pred.max().cpu().item(), 'delta': None, 'true_label': None})

        temp_df = pd.DataFrame(new_rows)
        pred_df = pd.concat([pred_df, temp_df], ignore_index=True)

    # Convert to numpy arrays for further processing
    features = np.array(features, dtype=np.float32)
    pseudo_labels = np.array(pseudo_labels, dtype=np.float32)

    if mode == 'segmentation':
        bins = [-np.inf, 0.1, 0.3, 0.5, 0.8, np.inf]
        pseudo_labels = pd.cut(pseudo_labels, bins, labels=[0, 1, 2, 3, 4]).astype(int)

    print(pred_df.shape)

    raise SystemExit

    # Flatten each feature
    N = features.shape[0]
    features = features.reshape(N, -1)

    # Standardize the features
    scaler = Normalizer()
    features = scaler.fit_transform(features)
    print(features.shape)


    # Use randomized solver in PCA for acceleration
    if features.shape[1] > 10e5:
        print(f'Old features.shape : {features.shape}')
        pca = PCA(n_components=min(features.shape[0], 1000), svd_solver='randomized')
        features = pca.fit_transform(features)
        print(f'New features.shape : {features.shape}')

    #endregion

    breakpoint()

    #region Dcom sample

    lSet = pred_df.loc[pred_df['true_label'].notna()].index.tolist()
    lSet_labels = pred_df.loc[lSet, 'true_label'].values.tolist()
    lSet_deltas = pred_df.loc[lSet, 'delta'].values.tolist()

    print(f'lSet : {lSet}')

    budgetSize = 6

    dcom = DCoM(features, lSet, budgetSize = budgetSize, lSet_deltas=lSet_deltas)
    active_set, new_uset, active_deltas = dcom.select_samples(pred_df)
    sampled_images = []


    for i, idx in enumerate(active_set):
        sampled_images.append(pred_df.iloc[int(idx)]['img_name'])
        pred_df.iloc[int(idx), pred_df.columns.get_loc('delta')] = active_deltas[i]

    print(f'2: {pred_df.loc[pred_df['delta'].notna()]}')

    # Save the sampled images to a new directory
    output_subdir = os.path.join(output_dir, 'samples')

    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir, exist_ok=True)

    print("Sampled image:")
    for img_name in sampled_images:
        img = os.path.join(data_path, img_name)
        output_path = os.path.join(output_subdir, img_name)
        shutil.copy(img, output_path)
        print(output_path)

    # Define the path to the labels subfolder
    labels_folder = os.path.join(output_dir, 'labels')

    if not os.path.exists(labels_folder):
        os.makedirs(labels_folder, exist_ok=True)

    # Determine the number of rows and columns for the grid
    num_images = len(sampled_images)
    num_cols = 5  # Number of columns
    num_rows = (num_images + num_cols - 1) // num_cols  # Calculate the number of rows needed

    # # Display the images in a grid layout
    # plt.figure(figsize=(20, 10))
    # for i, img_name in enumerate(sampled_images):
    #     img_path = os.path.join(data_path, img_name)
    #     try:
    #         img = Image.open(img_path)
    #         plt.subplot(num_rows, num_cols, i + 1)
    #         plt.imshow(img)
    #         plt.title(f"Image {i+1}, predicted label {pseudo_labels[active_set[i]]}")
    #         plt.axis('off')
    #     except Exception as e:
    #         print(f"Error loading image {img_path}: {e}")

    # plt.tight_layout()
    # plt.show()

    #endregion
    #region Dcom update D

    labels_folder = os.path.join(output_dir, 'labels')

    # Ensure only the images in the labeled set have a true label
    if 'true_label' in pred_df.columns:
        pred_df['true_label'] = np.nan


    # Read image names and labels from the labels subfolder
    for root, dirs, files in os.walk(labels_folder):
        for file_name in files:
            image_path = os.path.join(root, file_name)

            if mode == 'classification' and file_name.endswith('.jpg') :
                # Extract the label from the folder name
                parent_dir = os.path.dirname(root)
                folders = sorted(os.listdir(parent_dir))
                label = folders.index(os.path.basename(root))
                pred_df.loc[pred_df['img_name'] == file_name, 'true_label'] = label # Update the true_label column for the corresponding filename

            elif mode == 'segmentation' and file_name.endswith('.png'):
                img_name = os.path.splitext(file_name)[0] + '.jpg'
                # Calculate the proportion of white pixels
                image = Image.open(image_path).convert('L')  # Convert to grayscale
                image_array = np.array(image)
                white_pixel_count = np.sum(image_array == 255)
                total_pixel_count = image_array.size
                label = white_pixel_count / total_pixel_count
                pred_df.loc[pred_df['img_name'] == img_name, 'true_label'] = label

    if mode == 'segmentation' :
        pred_df['true_label'] = pd.cut(pred_df['true_label'], bins = bins, labels = [0, 1, 2, 3, 4])

    lSet = pred_df.loc[pred_df['true_label'].notna()].index.tolist()
    print(f'lSet : {lSet}')

    for i in list(set(pred_df.index)): #Ensures only the images in the labeled set have a delta
        if pred_df.at[i, 'true_label'] is np.nan:
            pred_df.at[i, 'delta'] = np.nan

    lSet_labels = pred_df.loc[lSet, 'true_label'].values.tolist()
    lSet_deltas = pred_df.loc[lSet, 'delta'].values.tolist()

    print(f'lSet : {lSet}')
    print(f'lSet_labels : {lSet_labels}')
    print(f'lSet_deltas : {lSet_deltas}')

    dcom = DCoM(features, lSet, budgetSize = budgetSize, lSet_deltas=lSet_deltas)
    lSet_deltas[-1 * budgetSize:] = dcom.new_centroids_deltas(lSet_labels, pseudo_labels=pseudo_labels, budget= budgetSize)

    # Update the deltas in the pred_df DataFrame
    pred_df.loc[lSet, 'delta'] = lSet_deltas

    #endregion
    #region plot

    # Visualize the features in 2D using t-SNE
    tsne = TSNE(n_components=2, random_state = 42)
    features_2d = tsne.fit_transform(features)

    # Extract the 2D coordinates for the sampled points
    sampled_points_2d = features_2d[lSet]

    plt.figure(figsize=(12, 6))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=pseudo_labels, cmap='viridis', s=30, label = 'Predicted pseudo-labels')
    # plt.scatter(sampled_points_2d[:, 0], sampled_points_2d[:, 1], c=pred_df.loc[pred_df['true_label'].notna(),'true_label'], cmap='viridis', s=31)
    # Add circles of radius delta around points from lSet
    ax = plt.gca()
    for i, point_idx in enumerate(lSet):
        point_2d = features_2d[point_idx]
        delta = float(lSet_deltas[i])
        circle = Circle(point_2d, radius=delta, color='blue', fill=False, linestyle='-', linewidth=0.5)
        ax.add_patch(circle)

    # Highlight the sampled points in a different color or marker (e.g., red stars)
    plt.scatter(sampled_points_2d[:, 0], sampled_points_2d[:, 1], c='red', marker='*', s=40, label='Sampled points')

    plt.colorbar()
    plt.title("Features Colored by Predicted Labels")

    # Add a legend to highlight the sampled points
    plt.legend()

    plt.show()


