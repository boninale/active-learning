'''
@article{hacohen2022active,
  title={Active learning on a budget: Opposite strategies suit high and low budgets},
  author={Hacohen, Guy and Dekel, Avihu and Weinshall, Daphna},
  journal={arXiv preprint arXiv:2202.02794},
  year={2022}
}

@article{mishal2024dcom,
      title={DCoM: Active Learning for All Learners}, 
      author={Mishal, Inbal and Weinshall, Daphna},
      journal={arXiv preprint arXiv:2407.01804},
      year={2024}
}

'''

import pandas as pd
import numpy as np
import torch
import random
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DCoM:
    """
    AL algorithm that selects the next centroids based on point density and model confidence, which measured
    using minimum margin.

    Main functions:
    - select_samples: Perform DCoM active sampling.
    - new_centroids_deltas: Perform binary search of the next delta values for the sampled set 
      in order to balance ball purity and radius(delta).
    """

    def __init__(self, features, lSet, budgetSize, lSet_deltas):

        print(f'\n Initializing DCoM algorithm.')

        self.features = features
        self.all_idx = list(features.keys())

        self.lSet = lSet
        self.uSet = list(set(self.all_idx) - set(self.lSet))

        self.lSet_deltas = lSet_deltas

        self.budgetSize = budgetSize
        if device.type == 'cuda':
            self.batch_size = 64
        else:
            self.batch_size = 8
        self.delta_resolution = 0.05 #Value used in initial paper
        self.k_logistic = 50 #Value used in initial paper
        self.a_logistic = 0.8 #Value used in initial paper


        self.relevant_indices = self.lSet + self.uSet # indices of lSet and then uSet in all_features
        self.rel_features = [self.features[k] for k in self.relevant_indices]  # features of lSet and then uSet

        if len(features) > 500:
            sample_keys = np.random.choice(self.all_idx, size=500, replace=False)
            sample_features = {k: self.features[k] for k in sample_keys}
        else:
            sample_features = self.features
                
        # Convert feature dictionaries to a list for pdist
        feature_matrix = np.array(list(sample_features.values()))
        self.max_delta = pdist(feature_matrix).max() * 2
        print(f'Max delta is {self.max_delta}')

        if not lSet_deltas:
            print('No deltas provided. Using initial delta value.')
            self.initial_delta = self.compute_initial_delta(feature_matrix)
            self.lSet_deltas = {k: self.initial_delta for k in self.lSet[:self.budgetSize]}
        else:
            self.initial_delta = self.compute_initial_delta(feature_matrix)
            for k, delta in zip(self.lSet, lSet_deltas):
                try:
                    delta = float(delta)
                    if delta <= 0:
                        raise ValueError
                except (ValueError, TypeError):
                    delta = self.initial_delta
                self.lSet_deltas[k] = delta

        self.delta_avg = np.average(list(self.lSet_deltas.values())) if self.lSet_deltas else self.initial_delta

        self.hash_to_idx = {h: i for i, h in enumerate(self.relevant_indices)}
        self.lSet_deltas_dict = {self.hash_to_idx[h]: delta for h, delta in self.lSet_deltas.items()}

    #             self.max_delta = pdist(sample_features).max() * 2
    #             print(f'Max delta is {self.max_delta}')

    #             if not lSet_deltas :
    #                 self.initial_delta = self.compute_initial_delta(sample_features)
    #                 self.lSet_deltas = [self.initial_delta for _ in range(self.budgetSize)]
    #             else:
    #                 self.lSet_deltas = []
    #                 self.initial_delta = self.compute_initial_delta(sample_features)
    #                 for i, delta in enumerate(lSet_deltas):
    #                     try:
    #                         delta = float(delta)
    #                         if delta <= 0:
    #                             raise ValueError
    #                     except (ValueError, TypeError):
    #                         print(f"Invalid delta at index {i} ({delta}). Using initial delta value.")
    #                         delta = self.initial_delta
    #                     self.lSet_deltas.append(delta)

    #             self.lSet_deltas_dict = dict(zip(np.arange(len(self.lSet)), self.lSet_deltas))
    #             self.delta_avg = np.average(self.lSet_deltas) if self.lSet_deltas else 0

    def compute_initial_delta(self, features):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        features_tensor = torch.tensor(features).to(device)
        distances = torch.cdist(features_tensor, features_tensor)
        delta0 = torch.median(distances).item()
        print(f'Initial delta is {delta0}')

        return delta0
    

    def construct_graph_excluding_lSet(self, delta=None, batch_size=64):
        """
        Creates a directed graph where:
        x -> y if l2(x, y) < delta.
        Considers all images, but does not reference or delete edges in lSet.

        The graph is represented by a list of edges (a sparse matrix) and stored in a DataFrame.
        """
        if delta is None:
            delta = self.delta_avg

        print(f'Start constructing graph using delta={delta}')
        # # Convert the list of NumPy arrays to a single NumPy array
        rel_features_array = np.array(self.rel_features)

        # Convert the NumPy array to a PyTorch tensor and move it to the specified device
        cuda_feats = torch.tensor(rel_features_array).to(device)  # All features on GPU

        # Prepare lists to store edges
        xs, ys, ds = [], [], []
        num_samples = cuda_feats.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size

        # Loop through batches
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            cur_feats = cuda_feats[start_idx:end_idx]  # Current batch

            # Compute distances between current batch and all features
            dist = torch.cdist(cur_feats, cuda_feats, p=2)  # p=2 for Euclidean distance
            mask = dist < delta  # Mask distances less than delta

            # Get indices of edges
            x, y = mask.nonzero().T
            
            # Offset batch index to global indices and store results
            xs.append(x + start_idx)  # Adjust x index by the batch start index
            ys.append(y)
            ds.append(dist[mask])

        # Concatenate and move results to CPU only once at the end
        xs = torch.cat(xs).cpu()
        ys = torch.cat(ys).cpu()
        ds = torch.cat(ds).cpu()

        # Create DataFrame from the final edge list
        df = pd.DataFrame({
            'x': xs.numpy(),
            'y': ys.numpy(),
            'd': ds.numpy()
        })

        print(f'Before delete lSet neighbors: Graph contains {len(df)} edges.')
        return df

    def construct_graph(self, delta=None, batch_size=64):
        """
         Creates a directed graph where:
         x -> y if l2(x, y) < delta, and deletes the covered points using lSet_deltas.

         Deletes edges to the covered samples (samples that are covered by lSet balls)
         and deletes all the edges from lSet.

         The graph is represented by a list of edges (a sparse matrix) and stored in a DataFrame.
         """
        
        if delta is None:
            delta = self.delta_avg

        df = self.construct_graph_excluding_lSet(delta, batch_size)

        # removing incoming edges to all cover from the existing labeled set
        edges_from_lSet_in_ball = np.isin(df.x, np.arange(len(self.lSet))) & (df.d < df.x.map(self.lSet_deltas_dict))
        covered_samples = df.y[edges_from_lSet_in_ball].unique()

        edges_to_covered_samples = np.isin(df.y, covered_samples)
        all_edges_from_lSet = np.isin(df.x, np.arange(len(self.lSet)))

        mask = all_edges_from_lSet | edges_to_covered_samples  # all the points inside the balls
        df_filtered = df[~mask]

        print(f'Finished constructing graph using delta={delta}')
        print(f'Graph contains {len(df_filtered)} edges.')
        return df_filtered, covered_samples

    def select_samples(self, confidences):
        """
        Performs DCoM active sampling section.
        """
        def get_competence_score(coverage):
            """
            Implementation of the logistic function weighting.
            """
            k = self.k_logistic  # the logistic growth rate or steepness of the curve
            a = self.a_logistic  # the logistic function center
            p = (1 + np.exp(-k * (1 - a)))
            competence_score = p / (1 + np.exp(-k * (coverage - a)))

            return round(competence_score, 3)

        print(f"\n==================== Start DCoM Active Sampling ====================")
        # Calculate the current coverage
        selected = []
        fully_graph = self.construct_graph_excluding_lSet(self.max_delta, self.batch_size)
        current_coverage = DCoM.calculate_coverage(fully_graph, self.lSet, self.lSet_deltas_dict, len(self.relevant_indices))
        del fully_graph

        competence_score = get_competence_score(current_coverage)

        margin = self.calculate_margin(confidences)
        margin[:len(self.lSet)] = [0] * len(self.lSet) # We define the margin score to be 1-margin (as described in our paper)

        cur_df, covered_samples = self.construct_graph(self.delta_avg, batch_size=self.batch_size)

        for i in range(self.budgetSize): # The active selection
            coverage = len(covered_samples) / len(self.relevant_indices)

            if len(cur_df) == 0:
                ranks = np.zeros(len(self.relevant_indices))
            else:
                # calculate density for each point
                ranks = self.calculate_density(cur_df)

            cur_selection = DCoM.normalize_and_maximize(ranks, margin, 1, lambda r, e: competence_score * e + (1 - competence_score) * r)[0]
            competence_score = get_competence_score(coverage)
            print(f'Iteration is {i}.\tGraph has {len(cur_df)} edges.\tCoverage is {coverage:.3f}. \tCurr choice is {cur_selection}. \tcompetence_score={competence_score}')

            new_covered_samples = cur_df.y[(cur_df.x == cur_selection)].values
            assert len(np.intersect1d(covered_samples, new_covered_samples)) == 0, 'all samples should be new'

            cur_df = cur_df[(~np.isin(cur_df.y, new_covered_samples))]  # Delete all the edges to the covered samples
            covered_samples = np.concatenate([covered_samples, new_covered_samples])
            margin[cur_selection] = 0
            selected.append(cur_selection)

        activeSet = [self.relevant_indices[i] for i in selected]
        remainSet = list(set(self.uSet) - set(activeSet))
        assert len(np.intersect1d(self.lSet, activeSet)) == 0, 'all samples should be new'
        for h in activeSet:
            self.lSet_deltas[h] = self.initial_delta
        print(f'Finished the selection of {len(activeSet)} samples.')
        return activeSet, remainSet, self.lSet_deltas

    def new_centroids_deltas(self, pseudo_labels,batch_size=64):
        """
        Optimized function for calculating delta values for centroids with dynamic adjustment for CPU and GPU.
        """
        print(f"\n==================== Updating lSet Deltas ====================")

        def calc_threshold(coverage):
            assert 0 <= coverage <= 1, f'coverage is not between 0 to 1: {coverage}'
            return 0.2 * coverage + 0.4

        # Set device for GPU or CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_gpu = device.type == "cuda"
        
        # Adjust batch size for CPU usage
        if not is_gpu:
            batch_size = min(batch_size, 32)  # Reduce batch size for CPU

        df = self.construct_graph_excluding_lSet(self.max_delta, self.batch_size)
        fully_df = self.construct_graph_excluding_lSet(self.max_delta, self.batch_size)

        # Prepare data on the correct device
        pseudo_labels_reordered = torch.tensor([pseudo_labels[h] for h in self.relevant_indices], dtype = int, device=device)
        df_x = torch.tensor(df.x.values, device=device)
        df_y = torch.tensor(df.y.values, device=device)
        df_d = torch.tensor(df.d.values, device=device)

        # Calculate the initial purity threshold based on coverage
        covered_samples = fully_df.y[
            (fully_df.x.isin(np.arange(len(self.lSet)))) & 
            (fully_df.d < fully_df.x.map(self.lSet_deltas_dict))
        ].unique()

        coverage = len(covered_samples) / len(self.relevant_indices)
        purity_threshold = calc_threshold(coverage)
        print("Initial purity threshold:", purity_threshold)

        # Initialize low and high delta values for binary search
        delta_resolution = self.delta_resolution  # Threshold for convergence in binary search

        # Helper function for batch purity check
        def check_purity_batch(cent_idx_batch, cent_labels_batch, deltas_batch):
            """
            Optimized purity check batch function with adjustments for CPU performance.
            """
            purities = torch.ones(len(cent_idx_batch), device=device)  # Default purity to 1

            # If on CPU, use a more memory-efficient approach
            if not is_gpu:
                for i, (cent_idx, cent_label, delta) in enumerate(zip(cent_idx_batch, cent_labels_batch, deltas_batch)):
                    neighbor_mask = (df_x.cpu().numpy() == cent_idx) & (df_d.cpu().numpy() < delta)
                    neighbors_idx = df_y[neighbor_mask] if len(neighbor_mask) > 0 else []

                    if len(neighbors_idx) > 0:
                        neighbors_pseudo_labels = pseudo_labels_reordered[neighbors_idx]
                        same_label_count = (neighbors_pseudo_labels == cent_label).sum().item()
                        purity = same_label_count / len(neighbors_idx)
                        purities[i] = purity
                    
            else:
                # GPU-specific vectorized operations
                for i, (cent_idx, cent_label, delta) in enumerate(zip(cent_idx_batch, cent_labels_batch, deltas_batch)):
                    neighbor_mask = (df_x == cent_idx) & (df_d < delta)
                    neighbors_idx = df_y[neighbor_mask]

                    if neighbors_idx.numel() > 0:
                        neighbors_pseudo_labels = pseudo_labels_reordered[neighbors_idx]
                        same_label_count = (neighbors_pseudo_labels == cent_label).sum().float()
                        purity = same_label_count / len(neighbors_idx)
                        purities[i] = purity

            return purities

        for batch_start in range(0, len(self.lSet), batch_size):
            batch_end = min(batch_start + batch_size, len(self.lSet))
            batch_centroids = self.lSet[batch_start:batch_end]
            batch_indices = [self.hash_to_idx[h] for h in batch_centroids]
            cent_labels_batch = [pseudo_labels[h] for h in batch_centroids]

            batch_low_deltas = torch.zeros(len(batch_centroids), device=device)
            batch_high_deltas = torch.full((len(batch_centroids),), self.max_delta, device=device)

            iteration = 0

            while torch.max(batch_high_deltas - batch_low_deltas) > delta_resolution:
                mid_deltas = (batch_low_deltas + batch_high_deltas) / 2

                purities = check_purity_batch(batch_indices, cent_labels_batch, mid_deltas)

                batch_high_deltas = torch.where(purities < purity_threshold, mid_deltas, batch_high_deltas)
                batch_low_deltas = torch.where(purities >= purity_threshold, mid_deltas, batch_low_deltas)

                iteration += 1

            # Assign final delta values
            final_deltas = (batch_low_deltas + batch_high_deltas) / 2
            for h, delta in zip(batch_centroids, final_deltas.cpu().numpy()):
                self.lSet_deltas[h] = delta

            print(f"Batch {batch_start // batch_size}/{len(self.lSet) // batch_size} completed in {iteration} iterations.")

        return self.lSet_deltas

    def calculate_density(self, df):
        counts = np.bincount(df['x'].values, minlength=len(self.relevant_indices))
        return counts

    def calculate_margin(self, confidences):
        print(f'Start calculating points margin.')
        ranks = -1 * np.array(list(confidences.values()))
        margin_result = ranks.reshape(-1, 1)
        scaler = MinMaxScaler()
        normalized_margin_result = scaler.fit_transform(margin_result).flatten()

        return normalized_margin_result.tolist()
    '''
    The original method used in the paper is the following:
    I changed it to use the confidence score in order to be able to use it with segmentation models as well. 
    The paper shows that the method used to evaluate confidence has little effect on the performance of the DCoM algorithm.

        def calculate_margin(self, model, train_data, data_obj):
            oldmode = model.training
            model.eval()

            print(f'Start calculating points margin.')
            all_images_idx = self.relevant_indices
            images_loader = data_obj.getSequentialDataLoader(indexes=all_images_idx,
                                                    batch_size=self.cfg.TRAIN.BATCH_SIZE, data=train_data)
            clf = model.cuda()
            ranks = []
            n_loader = len(images_loader)

            for i, (x_u, _) in enumerate(tqdm(images_loader, desc="All images Activations")):
                with torch.no_grad():
                    x_u = x_u.cuda(0)
                    temp_u_rank = torch.nn.functional.softmax(clf(x_u), dim=1)
                    temp_u_rank, _ = torch.sort(temp_u_rank, descending=True)
                    difference = temp_u_rank[:, 0] - temp_u_rank[:, 1]

                    # for code consistency across uncertainty, entropy methods i.e., picking datapoints with max value
                    difference = -1 * difference
                    ranks.append(difference.detach().cpu().numpy())
            ranks = np.concatenate(ranks, axis=0)
            print(f"u_ranks.shape: {ranks.shape}")

            model.train(oldmode)

            margin_result = np.array(ranks).reshape(-1, 1)
            scaler = MinMaxScaler()
            normalized_margin_result = scaler.fit_transform(margin_result)
            final_margin_result = np.array(normalized_margin_result.flatten().tolist())
            return final_margin_result
    '''


    @staticmethod
    def normalize_and_maximize(param1_list, param2_list, amount, target_func):
            """
            Perform pre-processing on each list and apply the target function on them.
            """
            param1_arr = np.array(param1_list).reshape(-1, 1)
            param2_arr = np.array(param2_list).reshape(-1, 1)

            # Min-Max normalization using scikit-learn's MinMaxScaler
            scaler = MinMaxScaler()
            param1_normalized = scaler.fit_transform(param1_arr)
            param2_normalized = scaler.fit_transform(param2_arr)

            # Calculate the product using the provided target_func
            product_array = target_func(param1_normalized.flatten(), param2_normalized.flatten())

            sorted_indices = np.argsort(product_array)[::-1]

            return sorted_indices[:amount]

    @staticmethod
    def calculate_coverage(fully_df, lSet, lSet_deltas_dict, total_data_len):
            """
            Return the current probability coverage.
            """
            covered_samples = fully_df.y[np.isin(fully_df.x, np.arange(len(lSet))) & (
                    fully_df.d < fully_df.x.map(lSet_deltas_dict))].unique()  # lSet send arrow to them
            return len(covered_samples) / total_data_len
