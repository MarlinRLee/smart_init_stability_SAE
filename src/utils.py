import torch
import numpy as np
import faiss
from tqdm import tqdm


def cosine_kmeans(dataloader, n_clusters, n_dims, max_samples=8_192_000, seed=42):
    """
    Computes Spherical K-Means centroids using Faiss on a sampled subset of data.
    
    Args:
        dataloader: Iterable yielding batches of tokens (e.g., [B, n_dims]).
        n_clusters: Number of centroids (32,000).
        n_dims: Dimension of tokens.
        max_samples: Number of vectors to use for training (Control memory usage here).
                      4M samples * 1024 dim * 4 bytes ~= 16 GB RAM.
    """
    print(f"--- Starting Faiss Spherical K-Means (k={n_clusters}) ---")
    print(f"Target Training Size: {max_samples} vectors")

    collected_samples = []
    current_count = 0
    
    print("Gathering training subset...")
    for batch in tqdm(dataloader, desc="Loading Data"):
        # Faiss requires float32 (standard float), convert from bfloat16/float16 if needed
        batch_np = batch.float().cpu().numpy()
        
        # If adding this batch exceeds max_samples, truncate it
        if current_count + len(batch_np) > max_samples:
            remaining = max_samples - current_count
            batch_np = batch_np[:remaining]
            collected_samples.append(batch_np)
            current_count += len(batch_np)
            break
        
        collected_samples.append(batch_np)
        current_count += len(batch_np)

    # 2. Prepare Data for Faiss
    # Concatenate and ensure C-contiguous array in RAM
    train_data = np.concatenate(collected_samples, axis=0)
    
    # L2 Normalize for Spherical K-Means (Cosine Similarity)
    # Faiss 'spherical=True' enforces normalization on centroids, 
    # but input data should also be normalized for correct cosine distances.
    faiss.normalize_L2(train_data)

    print(f"Data Shape: {train_data.shape} | Size: {train_data.nbytes / 1e9:.2f} GB")

    # --- 2. GPU Index Setup ---
    # We explicitly create a GPU resource and index. 
    # This guarantees the computation cannot happen on CPU.
    res = faiss.StandardGpuResources()
    
    # Configuration for the index
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False  # Set True if you run out of VRAM (40GB vs 80GB A100)
    
    # Create the index directly on the GPU
    # GpuIndexFlatIP = Inner Product (Cosine Similarity when normalized)
    index = faiss.GpuIndexFlatIP(res, n_dims, flat_config)

    # --- 3. Clustering Setup ---
    # Use the low-level Clustering object for fine-grained control
    clus = faiss.Clustering(n_dims, n_clusters)
    clus.spherical = True
    clus.niter = 20      
    clus.nredo = 3 
    clus.verbose = True
    clus.seed = seed

    # --- 4. Train ---
    print("Moving data to GPU and training...")
    
    # Train using the explicit GPU index
    clus.train(train_data, index)
    
    # --- 5. Extract Centroids ---
    # Centroids are stored in the Clustering object, usually on CPU after training
    centroids_np = faiss.vector_to_array(clus.centroids).reshape(n_clusters, n_dims)
    
    # Convert to torch and normalize one last time
    centroids = torch.from_numpy(centroids_np).float()
    centroids = torch.nn.functional.normalize(centroids, p=2, dim=1)
    
    return centroids