# 1. What is a Cluster?
# A cluster is a group of data points that are more similar to each other than 
#to those in other groups. Clustering is a way of finding patterns in data 
#without knowing the actual class labels.

# 2. Application of Clustering
# - Marketing: Segmenting customers based on purchasing behavior.
# - Healthcare: Grouping patients with similar health conditions.
# - Image Processing: Grouping pixels based on color for segmentation.
# - Anomaly Detection: Identifying outliers or rare events.

# 3. Types of Clustering
# - K-Means Clustering: A centroid-based algorithm where data points are 
#assigned to K clusters based on proximity to cluster centroids.
# - Hierarchical Clustering: A tree-based algorithm that groups similar items 
#together into a hierarchy.
# - DBSCAN (Density-Based Clustering): Groups dense regions of points, effective
#for noise and non-spherical shapes.

# 4. K-Means Clustering on the Iris Dataset
# - We'll use the Iris Dataset, which is built into R. The dataset contains 150 
#observations across 5 attributes: Sepal Length, Sepal Width, Petal Length, 
#Petal Width, and Species.

# 1. Load Libraries:
# Load necessary libraries
library(ggplot2)  # For visualization
library(cluster)  # For clustering and silhouette

# 2. Load the Iris Dataset:
# Load the built-in Iris dataset
data(iris)
head(iris)  # View the first few rows of the dataset

# 3. Prepare the Data:
# We'll exclude the Species column since it's categorical and not needed for 
#clustering.

# Select only the numeric columns (excluding 'Species')
iris_numeric <- iris[, 1:4]
head(iris_numeric)  # View the numeric data

# 4. K-Means Clustering:
# Set seed for reproducibility
set.seed(123)

# Apply K-Means clustering (let's choose K = 3 since there are 3 species in the dataset)
kmeans_model <- kmeans(iris_numeric, centers = 3)

# View the cluster assignments
kmeans_model$cluster

# Add the cluster assignments to the original dataset
iris$Cluster <- as.factor(kmeans_model$cluster)

# View the updated dataset with cluster assignments
head(iris)

# 5. Visualize K-Means Clusters:
# Visualize the clusters using ggplot2
ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Cluster)) +
  geom_point() +
  labs(title = "K-Means Clustering on Iris Dataset", x = "Sepal Length", y = "Sepal Width")

# 5. Hierarchical Clustering (Dendrogram)
# Now let's apply Hierarchical Clustering and plot the Dendrogram.

# 1. Compute Distance Matrix:
# Compute the distance matrix using Euclidean distance
dist_matrix <- dist(iris_numeric)

# 2. Apply Hierarchical Clustering:
# Apply Hierarchical clustering using Ward's method
hclust_model <- hclust(dist_matrix, method = "ward.D2")

# Plot the dendrogram
plot(hclust_model, main = "Dendrogram for Hierarchical Clustering", xlab = "", sub = "")

# 6. Cluster Validation
# We'll use two common methods to validate the clusters:
# - Elbow Method (for K-Means)
# - Silhouette Score (for both K-Means and Hierarchical Clustering)
# Elbow Method:
# - This method helps us determine the optimal number of clusters by looking for the "elbow" point where the within-cluster sum of squares decreases at a slower rate.

# Calculate the total within-cluster sum of squares (WSS) for different values of K
wss <- sapply(1:10, function(k) kmeans(iris_numeric, centers = k)$tot.withinss)

# Plot the WSS values
plot(1:10, wss, type = "b", pch = 19, frame = FALSE, 
     xlab = "Number of Clusters", ylab = "Total Within-Cluster Sum of Squares",
     main = "Elbow Method for Optimal K")

# Silhouette Score:
# - The Silhouette Score measures how similar points are to their own cluster 
#compared to other clusters. A higher score indicates better-defined clusters.

# Compute the silhouette score for K-Means
silhouette_values <- silhouette(kmeans_model$cluster, dist_matrix)

# Plot the silhouette scores
plot(silhouette_values, main = "Silhouette Plot for K-Means Clustering")



