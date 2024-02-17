# Data clustering using MeanShift and KMeans
This project is an example of data clustering using `MeanShift` and `KMeans` algorithms from the `scikit-learn` library. Here we create and visualize artificial data and then apply both algorithms to that data for clustering.

## Description

This code implements the `MeanShift` and `KMeans` clustering algorithm for a dataset created using the `make_blobs` function from the `scikit-learn` library. It explores the structure of the data and finds the optimal number of clusters using the silhouette metric (`silhouette_score`), and then visualizes the clustering results using graphs.

This can be useful for analyzing data, identifying groups or patterns in data such as user segmentation, market analysis, identifying product groups and other structures in data.

- <strong>Data Generation:</strong> We start by generating artificial data using the make_blobs function from the scikit-learn library. This function creates clusters of data points with given centers and standard deviation.

- <strong>Clustering with `MeanShift`:</strong> We apply the `MeanShift` algorithm to our data for clustering. The algorithm automatically determines the number of clusters.

- <strong>Visualization of raw data and cluster centers by `MeanShift`:</strong> We visualize the raw data and cluster centers determined by the MeanShift algorithm.

- <strong>Determining the optimal number of clusters using `KMeans`:</strong> We use the `KMeans` algorithm to determine the optimal number of clusters based on the `silhouette_score` metric.

- <strong>Clustering with `KMeans`:</strong> We apply the `KMeans` algorithm with the optimal number of clusters to our data.

- <strong>Visualization of clustered data with clustering regions:</strong> We visualize clustered data and cluster centers using `KMeans`.

## Screenshots
  
<table><tr>
  <td colspan="2">
  <img src="https://i.imgur.com/flNgMGx.png" alt="Image" width="900" height="500">

</td>
</tr>
<tr>
<td>
  <img src="https://i.imgur.com/jkYvII2.png" alt="Image" width="450" height="495">
</td>
  <td><img src="https://i.imgur.com/IJOAmHi.png" alt="Image" width="450" height="495"></td>
</tr>
</table>

## Requirements

To successfully deploy and run the project, you will need the following libraries listed in the `requirements.txt` file:

- matplotlib
- scikit-learn

## Installation

1. Clone the project repository to your local computer:
    ```bash
    git clone https://github.com/Dogherty/ClusterAnalysis.git
    
      
3. Install the Python dependencies specified in requirements.txt:
     ```bash
       pip install -r requirements.txt
