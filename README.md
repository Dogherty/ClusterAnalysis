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
  <img src="https://lh3.googleusercontent.com/pw/ABLVV85QDr2pI-OLA9wVMCECLnjJ6po4FzgOltctlKeycKWr8xkgRzEKVijI7agfUGhhJvN2DlsLYDXuo9SYlAxBQl7-0lOB-jxGiDgFfdSjDGI4XTNajuB2lGw1SG3fmn4Wd97ATnYJVyoliZA_C-wCZq53nmzTu3Oudvw0fpTPADhdBk-_u4owPJYWXfPjm1Z59iyNRtnZYgQMdF2AejCh1JftP1_faZQIdABbnMM5NXaOmtL3-ZU-du7QwPmh4dR58F1oP3rpUtPPZ39mz_hjIPJ9DMrzGDJKXcnZqDLsK3HEj61A3yv0AbFSxKN5PDIqo9TnDmL2Y6JLMh_C0L8aTSLUnstzVwvwbDFMWstHQHUHQNur8veFUOqmkvEv28RHI-ausy9ahlV2nR_OsuTLUPOWb7e-FFlrElUSXfk9ROwxcw27im0eNuHwdVUBAge9sCO9RgLDra7hi779R0Fv7MD3zxbHaxLXxJ1qHnDc1DJkcWXnqRqJwlAcUAowL5T6nGPV83BRnuprfpB1OjiBmmleMv_HZikByXMEOpyXhwbZnrzwZff29cN1Ytg2dIMP-2bWqPtRdp5e9PCN8DrbeCwL8wBWSOYno48alzhMF8Hbtnv736cuaJ8a5o6h_8X0KEYwyf1vuFqnWTK_iuvM0z-JEXsRMCQlhZMyZxiuwBGeX57kSfVvP4Qv5FYVN5j_kOjfQSGzHDvxi4dX9b0WRQxv7HZPEKln6NJcD1_SUfRjWUQLaPMXRmnIZMtq2nDokFwc3M4knrhmreYPhHvvbnoGz2JAp4SR_sZj_oLSADRfedWQOxkbhTNTsBKQ3n8Lsi2opB5x28jsE-xhFZoNSfZc-oerWlfKn9XKVV-Y7tnzb0Bm_W-JrdwlfAHEeekOE7wSkSELlVrh68aN-g1Cc44AgoM=w1198-h670-s-no?authuser=0" alt="Image" width="900" height="500">

</td>
</tr>
<tr>
<td>
  <img src="https://lh3.googleusercontent.com/pw/ABLVV84yOfF5CrmDbFtfXkn8fuBEOSdAvN-fBxohCaiF3zrXaIlSAjFLzijGNA2As6th1XJglVH9g3o6BAtGBjhdcLvRvqy9-XZxdHQx7Br_ZZRwMaGHV7L4OrOENnvdYMuv3TD4bDJTaLTw_yTSUajJ_YbZiF8HD1ChRSAeLyDOlJ1QmEn7vqtakOW5pj9u30cYL_59D9iST69gEbKgf7tNSyOz0WIyf46FyPbjt9opzlA1cW7aL_TgfpXa8dryld9gfSr2fbPJTNg6INSfibcvmlqzCRqm_VbSuiEc4_tPPA0wHByo06JrbQTH4eNuMLlfB41TPfkvmcKeuJoT8BXmZZeR7zBviI41GF-o_74JWihueJG6wyU5vET2QLUaZIolx8zSUduORIuQtd17zAY8w8PRDc8XMFkwlfFLo_G01BI1l-q2-JiYe3jI2g4Ye5b78HkROJYAQ_qK1GGPUzCNjmFyUVDemFGOQPubIpXQSlNoK6MucJQ4iowL78ipccsrc-XIN5zKYIyCLhnQzmYxvQd2tfPi1uOpiY23ETsa9XulciIF4IzLwydj3NRggq7YwEqbLWdAPcWamcHJ1U2LR7Je5mVeqNt7Hn80iuL6Mqz__npyGYvxkShmoR35BhccfWRd2zvkvCYnA-zxD9teEkt_A3jaETUxnhW7rL-gPMjft62e-UlnFmYcbU-wN5xliIKiBIUzCR8cfBH5BZQi02eY0XQ1-Ms6qrtH5lWXK42cIrF3K8QtqETN4KNYxG_LUg9oCInbI8cXU5S8kRg3rIP_fO85A48GvzCHJPGv9nsqxfMZ_9ECRwIHTnRsnQLZCfSpnHBx3PQlCsTLgh8fOXUstzLxOIgUH8WnNWsiewGYRaLd0bqVqDyi-J08UXcCRoiQ_NdQxofU_-DiJXqqaqNTTYM=w598-h671-s-no?authuser=0" alt="Image" width="450" height="495">
</td>
  <td><img src="https://lh3.googleusercontent.com/pw/ABLVV87Keh8BTwNaNVBmRAwJ5BnSrVOF5EU1gTH8XRAp7MmjL7jsDHR3ayC78H1nU_LNtTvIjKF_CiaJrMCM18mYh2vwLZ3vN0zFEXEmz1i_QrnaVsXaoeKzHNOJRYGO9EwPhC9zthr3tT8j43Lc3icQa_qCiiGgM9s-OUkZ_8BewKUccHPwT8hzLUu9ls2x0ugW2ber8N4afLwjZ4Vv5bdfkVBt1zJqmbzyh7VnyO-D2JS9PGLA8X3L_Y9xriiGXKdxu-u8x5LAJQ3BRWb7s2WF_P6O4hzRTV7Bnt64Rk5k6FQGEl0ZcM_OApR9iNCtAemAqggZzW19KtynmD3Obuo2gys7D_ixAKEdtFdWheDdt9JPp0XFH9WZTB4S-BCCwXkRH_TKvkf-FZ6tY1kvkxUhoJwGPSZHbm-GfYTS8Dfi9S5F3KOD_riD_OP6h0Lv617dP9DGurlV3z5eN8PBX0sbQuRAW9YFfziJJ0u-M82SnKMy0Yl7nhUHlz7kUpfWjuMEbUI36xafyLx6E4GPLugGSz4IhbAGCh-BhAMB0fYBnbMo-N9f9ixhiSUxwTiUjNSg62MjTtwd4meCL1cNiv4lwE1LJ4H1i-O6C61wbTNCemrgf4G6aDIzRFJmqHU9cNTzMcBKwv1F230Q-GDrJqbybCH2tcQQDsCRlQiaAi4TvTu1SXvKCx9ctaE7PN9zNkQTRXRjmUzB6t6GM5hBydscsXIJZ9fK8fSqLozxd7f98Y6RFddDyWgcJmoZwj7S7H48llBdOuQmYCoZgaVTh0EWBoz-6xu65In6Zu25PSeHmeXu0DnEMYgT7ijBUXIdUxCmv2G8yyXUNVo9xN-cEsdZ1J32f9yXJOVA_hVHAWvCTykYCbBkHAovjT4H8l7RXUc7uH5tdvlYi2csOKgXOGvlqDJTk-4=w598-h671-s-no?authuser=0" alt="Image" width="450" height="495"></td>
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
