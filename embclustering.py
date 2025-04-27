import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from dotenv import load_dotenv
import openai


# Load environment variables
load_dotenv()
script_dir = os.path.dirname(os.path.abspath(__file__))

# Intialize OpenAI API client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Read list of insurance attributes from a file
def load_insurance_attributes(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

insurance_attributes_file = os.path.join(script_dir, "insurance_attributes.txt")
insurance_attributes = load_insurance_attributes(insurance_attributes_file)

# Function to get embeddings in batches
def get_embeddings(texts, batch_size=100):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            input=batch,
            model="text-embedding-3-small"
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    return np.array(all_embeddings)

# Function to visualize clusters using TSNE
def visualize_clusters_tsne(embeddings, labels, texts, fig_size=(16, 10)):
    # Reduce dimensions to 2D
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=fig_size)
    
    # Get unique labels and generate colors
    unique_labels = set(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    # Plot each cluster
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise
            col = 'k'
        
        class_member_mask = (labels == k)
        xy = reduced_embeddings[class_member_mask]
        
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, 
                 markeredgecolor='k', markersize=6)
    
    # Add some representative text labels
    for cluster_id in [label for label in unique_labels if label != -1]:
        # Get indices of this cluster
        indices = np.where(labels == cluster_id)[0]
        
        # Find point closest to centroid
        centroid = np.mean(reduced_embeddings[indices], axis=0)
        closest_idx = indices[np.argmin(np.sum((reduced_embeddings[indices] - centroid)**2, axis=1))]
        
        # Add text label
        plt.annotate(f"Cluster {cluster_id}: {texts[closest_idx][:30]}...", 
                     xy=reduced_embeddings[closest_idx], 
                     xytext=(5, 2), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    plt.title('t-SNE Visualization of Insurance Attribute Clusters')
    plt.savefig(os.path.join(script_dir, 'insurance_clusters_tsne.png'))
    plt.show()

# Function to visualize clusters using PCA (alternative to TSNE)
def visualize_clusters_pca(embeddings, labels, texts, fig_size=(16, 10)):
    # Reduce dimensions to 2D
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=fig_size)
    
    # Get unique labels and generate colors
    unique_labels = set(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    # Plot each cluster
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise
            col = 'k'
        
        class_member_mask = (labels == k)
        xy = reduced_embeddings[class_member_mask]
        
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, 
                 markeredgecolor='k', markersize=6)
    
    # Add some representative text labels for all clusters
    for cluster_id in [label for label in unique_labels if label != -1]:
        # Get indices of this cluster
        indices = np.where(labels == cluster_id)[0]
        
        # Find point closest to centroid
        centroid = np.mean(reduced_embeddings[indices], axis=0)
        closest_idx = indices[np.argmin(np.sum((reduced_embeddings[indices] - centroid)**2, axis=1))]
        
        # Add text label
        plt.annotate(f"Cluster {cluster_id}: {texts[closest_idx][:30]}...", 
                     xy=reduced_embeddings[closest_idx], 
                     xytext=(5, 2), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    # Show explained variance
    explained_variance = pca.explained_variance_ratio_
    plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
    plt.title('PCA Visualization of Insurance Attribute Clusters')
    plt.savefig(os.path.join(script_dir, 'insurance_clusters_pca.png'))
    plt.show()

# Function to find the largest cluster and its centroid
def get_largest_cluster_info(embeddings, labels, texts):
    unique_labels = np.unique(labels)
    # Exclude noise points (label -1)
    cluster_labels = [l for l in unique_labels if l != -1]
    
    if not cluster_labels:
        return None, None, None
    
    # Count sizes of each cluster
    cluster_sizes = [(label, np.sum(labels == label)) for label in cluster_labels]
    # Sort by size (descending)
    sorted_clusters = sorted(cluster_sizes, key=lambda x: x[1], reverse=True)
    
    # Get the largest cluster label
    largest_cluster_label = sorted_clusters[0][0]
    largest_cluster_size = sorted_clusters[0][1]
    
    # Get indices of the largest cluster
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
    
    # Calculate centroid embedding
    centroid_embedding = np.mean(embeddings[largest_cluster_indices], axis=0)
    
    # Find the item closest to the centroid
    distances = np.sum((embeddings[largest_cluster_indices] - centroid_embedding)**2, axis=1)
    closest_idx = largest_cluster_indices[np.argmin(distances)]
    centroid_text = texts[closest_idx]
    
    return largest_cluster_label, centroid_embedding, centroid_text

# Function to analyze and print cluster statistics
def analyze_clusters(labels, texts):
    unique_labels = np.unique(labels)
    cluster_count = len([l for l in unique_labels if l != -1])
    noise_count = np.sum(labels == -1)
    
    print(f"Total clusters found: {cluster_count}")
    print(f"Points classified as noise: {noise_count} ({noise_count/len(labels):.1%} of total)")
    
    # Get sizes of each cluster
    cluster_sizes = [(label, np.sum(labels == label)) for label in unique_labels if label != -1]
    sorted_clusters = sorted(cluster_sizes, key=lambda x: x[1], reverse=True)
    
    print("\nTop clusters by size:")
    for i, (label, size) in enumerate(sorted_clusters[:10]):
        print(f"Cluster #{i+1} (ID: {label}): {size} items ({size/len(labels):.1%} of total)")
        
        # Show representative examples
        examples_idx = np.where(labels == label)[0][:3]
        print("Examples:")
        for idx in examples_idx:
            print(f"  - {texts[idx]}")
        print()

# Function to try different DBSCAN parameters and evaluate clustering quality
def find_optimal_dbscan_params(embeddings, texts):
    # Define parameter grid to search - with much larger eps values
    eps_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    min_samples_values = [2, 3, 4, 5]
    
    best_params = (0.3, 3)  # Default fallback if no clusters are found
    best_score = -1
    best_labels = None
    
    # Simple scoring function: more clusters with less noise is better
    # but we penalize having too many clusters or too much noise
    def score_clustering(labels):
        unique_labels = np.unique(labels)
        cluster_count = len([l for l in unique_labels if l != -1])
        noise_count = np.sum(labels == -1)
        total_count = len(labels)
        
        # No clusters is bad
        if cluster_count == 0:
            return -1
        
        # Too many clusters is also bad (more than 30% of points being their own cluster)
        if cluster_count > total_count * 0.3:
            return -1
        
        # Too much noise is bad (more than 70% noise)
        if noise_count > total_count * 0.7:
            return -1
        
        # Score based on cluster count and reduced noise
        score = cluster_count * (1 - noise_count/total_count)
        return score
    
    print("Finding optimal DBSCAN parameters...")
    for eps in eps_values:
        for min_samples in min_samples_values:
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
            labels = clustering.labels_
            
            score = score_clustering(labels)
            
            # Count clusters and noise points
            unique_labels = np.unique(labels)
            cluster_count = len([l for l in unique_labels if l != -1])
            noise_count = np.sum(labels == -1)
            
            print(f"eps={eps}, min_samples={min_samples}: {cluster_count} clusters, {noise_count} noise points, score={score:.2f}")
            
            if score > best_score:
                best_score = score
                best_params = (eps, min_samples)
                best_labels = labels
    
    # If no good parameters found (best_score still -1), use hardcoded values
    if best_score == -1:
        print("\nNo optimal parameters found, using default eps=0.4, min_samples=2")
        clustering = DBSCAN(eps=0.4, min_samples=2).fit(embeddings)
        best_labels = clustering.labels_
        best_params = (0.4, 2)
    else:
        print(f"\nBest parameters: eps={best_params[0]}, min_samples={best_params[1]}, score={best_score:.2f}")
    
    return best_params, best_labels

# Main execution
def main():
    print("Generating embeddings...")
    embeddings = get_embeddings(insurance_attributes)
    
    # Find optimal DBSCAN parameters
    (eps, min_samples), best_labels = find_optimal_dbscan_params(embeddings, insurance_attributes)
    
    print(f"\nUsing DBSCAN with eps={eps}, min_samples={min_samples}")
    
    # Analyze clusters
    analyze_clusters(best_labels, insurance_attributes)
    
    # Get largest cluster info
    largest_cluster_label, centroid_embedding, centroid_text = get_largest_cluster_info(
        embeddings, best_labels, insurance_attributes)
    
    if largest_cluster_label is not None:
        print("\n--- LARGEST CLUSTER INFORMATION ---")
        print(f"Largest Cluster Number: {largest_cluster_label}")
        print(f"Centroid Text Representation: {centroid_text}")
        print(f"Centroid Embedding (first 10 dimensions): {centroid_embedding[:10]}...")
    else:
        print("\nNo clusters were found with the current parameters.")
    
    # Always visualize using both methods regardless of dataset size
    print("\nGenerating visualizations...")
    print("Generating t-SNE visualization...")
    visualize_clusters_tsne(embeddings, best_labels, insurance_attributes)
    
    print("Generating PCA visualization...")
    visualize_clusters_pca(embeddings, best_labels, insurance_attributes)

if __name__ == "__main__":
    main()