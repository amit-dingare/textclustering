# Insurance Attribute Clustering

A Python tool that uses embedding-based clustering to organize insurance attributes into meaningful groups based on semantic similarity. This tool leverages OpenAI's embedding model to transform text attributes into high-dimensional vectors, then applies DBSCAN clustering to identify natural groupings in the data.

## Features

- Generates text embeddings using OpenAI's text-embedding-3-small model
- Automatically finds optimal DBSCAN clustering parameters
- Visualizes clusters using both t-SNE and PCA dimensionality reduction
- Identifies the largest cluster and its centroid representation
- Provides detailed statistics and examples for each cluster

## Requirements

- Python 3.6+
- OpenAI API key
- Required Python packages:
  - numpy
  - matplotlib
  - scikit-learn
  - openai
  - python-dotenv

## Installation

1. Clone this repository:
2. Install the required packages:
3. Change the `env-example` to `.env` file in the root directory and add your OpenAI API key:

## Usage

The script will:
1. Load insurance attributes from `insurance_attributes.txt`
2. Generate embeddings for each attribute using OpenAI's API
3. Find optimal DBSCAN parameters by testing multiple eps/min_samples combinations
4. Cluster the attributes and analyze the results
5. Identify the largest cluster and its centroid representation
6. Generate and save visualization plots:
   - `insurance_clusters_tsne.png`: t-SNE visualization
   - `insurance_clusters_pca.png`: PCA visualization

## How It Works

### Embedding Generation

The script converts each text attribute into a high-dimensional vector (embedding) that captures its semantic meaning using OpenAI's text-embedding-3-small model.

### Parameter Optimization

The script tests multiple combinations of DBSCAN parameters (eps and min_samples) to find the optimal settings that create meaningful clusters with minimal noise.

### Clustering

Using the optimal parameters, DBSCAN identifies natural groupings in the embedding space:
- Points that are close together (within eps distance) are grouped into clusters
- Points that don't have enough neighbors (min_samples) are labeled as noise (-1)

### Visualization

The high-dimensional embeddings are projected into 2D space using:
- t-SNE: Preserves local relationships between points
- PCA: Preserves global variance in the data

### Analysis

For each cluster, the script:
- Calculates its size and percentage of the total dataset
- Shows representative examples
- Identifies the centroid (average) embedding
- Finds the text attribute closest to the centroid

## Customization

You can modify the script to adjust:
- The range of DBSCAN parameters tested in `find_optimal_dbscan_params()`
- The scoring function to prioritize different clustering qualities
- The visualization settings (figure size, colors, annotations)

## Troubleshooting

If no clusters are found:
- Try expanding the range of eps values (higher for more spread-out data)
- Reduce the min_samples requirement
- Consider using a different clustering algorithm (K-means, Agglomerative)

If too many points are classified as noise:
- Increase the eps parameter to form larger clusters
- Decrease the min_samples parameter to form clusters with fewer points

## License

[Specify the license information here]

## Contributing

[Guidelines for contributing to the project]