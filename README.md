# Active Learning Script Documentation

This script implements an active learning framework for image datasets. It includes functionality for loading data, extracting features, performing active learning using the DCoM algorithm, and visualizing the results. 
<br/>
The DCom method is a state-of-the-art method for active learning (Cf. Paper with code ranking) achieving high performance in both the low and high budget regimes.
<br/>
According to the authors of the algorithm : "*DCoM employs a representation learning approach. Initially, a 
Δavg-radius ball is placed around each point. The Δlist provides a specific radius for each labeled example individually. From these, a subset of b balls is chosen based on their coverage of the most points, with the centers of these balls selected as the samples to be labeled. After training the model, the Δ list is updated according to the purity of the balls to achieve more accurate radii and coverage. DCoM utilizes this coverage to determine the competence score, which balances typicality and uncertainty.*"
<br/> 

The algorithm is based on the paper Mishal, Inbal, et Daphna Weinshall. « DCoM: Active Learning for All Learners ». arXiv, 24 juillet 2024. http://arxiv.org/abs/2407.01804.

## Table of Contents

- Setup
- Usage
- Parameters
- Functions
- Example
- Output

## Setup

1. **Clone the Repository:**
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Install Dependencies:**
    Install the required Python packages using `pip`:
    ```sh
    pip install -r requirements.txt
    ```

3. **Prepare the Dataset:**
    Ensure that your dataset is organized in the specified directory structure and that the paths are correctly set in the script.

## Usage

1. **Set Parameters:**
    Modify the parameters in the script to match your dataset and requirements. Key parameters include:
    - 

mode

: Set to `'classification'` or `'segmentation'`.
    - 

data_path

: Path to the dataset directory.
    - 

model_path

: Path to the pre-trained model.
    - 

budgetSize

: Number of samples to select in each active learning iteration.
    - 

batch_size

: Batch size for data loading.

2. **Run the Script:**
    Execute the script to start the active learning process:
    ```sh
    python active_learning.py
   ```
3. **At the breakpoint**
    Go the dataset, the chosen samples will be saved to a subfolder called 'DCoM'. Label the samples (for classification put them in the right class folders and for segmentation create the segmentation masks) and att the labels to the 'labels' folder. This step is crutial for the algorithm to work

## Parameters

- **mode**: Specifies the mode of operation (`'classification'` or `'segmentation'`).
- **data_path**: Path to the directory containing the dataset.
- **model_path**: Path to the pre-trained model file.
- **budgetSize**: Number of samples to select in each active learning iteration.
- **batch_size**: Batch size for data loading.
- **save_pred_df**: Boolean flag to save the predictions DataFrame.
- **save_samples**: Boolean flag to save the sampled images.

## Functions

- **ImageDataset**: Custom Dataset class for loading and preprocessing images.
- **custom_collate**: Custom collate function to handle batches with `None` values.
- **DCoM**: Class implementing the DCoM active learning algorithm.

## Example

```python
# Set parameters
mode = 'classification'
data_path = r'/path/to/dataset'
model_path = r'/path/to/model.pth'
budgetSize = 1000
batch_size = 128
save_pred_df = True
save_samples = True

# Run the script
if __name__ == '__main__':
    # Feature extraction and active learning process
    ...
```

## Output

- **features.pkl**: Pickle file containing extracted features from the dataset.
- **pred_df.pkl**: Pickle file containing predictions and other metadata for the dataset.
- **pseudo_labels.pkl**: Pickle file containing pseudo labels for the dataset.
- **features_plot.png**: Plot visualizing the features in 2D using t-SNE.



## Detailed Steps

1. **Initialization:**
    - Set the random seed for reproducibility.
    - Define the device (CPU or GPU) for computation.
    - Load the pre-trained model.

2. **Feature Extraction:**
    - Iterate over the images in the dataset.
    - Extract features and predictions using the model.
    - Store the features and predictions in a DataFrame.

3. **Active Learning with DCoM:**
    - Initialize the DCoM algorithm with the extracted features and labeled set.
    - Select samples using the DCoM algorithm.
    - (optional) Save the samples with their pseudo labels in the 'DCoM/samples subdirectory' of the dataset.
    - Update the predictions DataFrame with the selected samples.

4. **Visualization:**
    - Visualize the features in 2D using t-SNE.
    - Highlight the sampled points and labeled points in the plot.
    - Save the plot to the output directory.

5. **Save Results:**
    - Save the predictions DataFrame, features, and pseudo labels to pickle files.

## Notes

- Ensure that the dataset directory and model path are correctly set before running the script.
- The script uses multiprocessing for data loading to optimize performance. Adjust the 

num_workers

 parameter based on your system's capabilities.
- The script includes a breakpoint to allow for manual labeling of images before continuing with the active learning process.

By following this documentation, you should be able to set up and run the active learning script effectively. If you encounter any issues or have questions, please refer to the comments in the script or contact the project maintainers.