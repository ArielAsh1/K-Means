
import matplotlib.pyplot as plt
import numpy as np

# Generate a 2D dataset
X = np.concatenate([
    np.random.normal([0, 0], size=(500, 2)),
    np.random.normal([5, 5], size=(500, 2)),
    np.random.normal([5, 0], size=(500, 2)),
    np.random.normal([0, 5], size=(500, 2)),
])

# Shuffle the data
np.random.shuffle(X)
print(X.shape)

# Ploting the data to see it's clusters
plt.scatter(X[:, 0], X[:, 1], cmap='viridis')

# the actual K-Means algorithm

class KMeans():
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

        ######## Helper fields #########
        # stores the dataset X.
        self.X_fit_ = None
        # stores the final labels. That is, the clusters indices for all the samples
        self.labels_ = None
        # stores the final centroids.
        self.centroids = None
        # stores the labels of each iteration.
        self.labels_history = []
        # stores the centroids of each iteration.
        self.centroids_history = []
        # stores the costs of the iterations. we calculate the cost in every iteration and store it in this list.
        self.costs = []

    def fit(self, X):
        self.X_fit_ = X
        # get random prototypes (=centroids)
        self.centroids = self.X_fit_[np.random.choice(len(self.X_fit_), self.n_clusters, replace=False)]

        t = 0  # iteration counter
        while t != self.max_iter:
            # add current centroids to keep track of used centroids
            self.centroids_history.append(self.centroids)
            # keep track of cost for each iteration
            self.costs.append(self._calculate_cost(self.X_fit_))
            # assign each sample to label according to its closest centroid
            self.labels_ = self.predict(self.X_fit_)
            self.labels_history.append(self._get_labels(self.X_fit_))
            # calculate a new centroid for each cluster according to mean between all samples clusterd to that cluster
            new_centroids = np.array(
                [X[self._get_labels(self.X_fit_) == i].mean(axis=0) for i in range(self.n_clusters)])
            # check if the centroids haven't changed, then algorithm has converged
            if np.all(self.centroids == new_centroids):
                break

            # if centroids changed then update centroids
            self.centroids = new_centroids

            t += 1

    def predict(self, X):
        return np.argmin(self._get_distances(X), axis=1)

    def _get_distances(self, X):
        #  L2 norm (Euclidean distance)
        return np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

    def _get_labels(self, X):
        return self.labels_

    def _get_centroids(self, X, labels):
        return self.centroids

    def _calculate_cost(self, X):
        # calculate D learned in class where D is the total distance over dataset
        # from a sample to it's centroid
        return np.sum(np.min(self._get_distances(X), axis=1))

# Run the algorithm on the 2D dataset
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X)
# and plot a graph of the costs as a function of the iterations
costs = kmeans.costs
plt.plot(costs)

""" the following are the results of the algorithm's runs with n_clusters = 2, 3, 4, 6, 8, 10, 20
 and the final cost  in each n_clusters.
n_clusters = 2 --> 5405.765897311589
n_clusters = 3 --> 3873.9310656015423
n_clusters = 4 --> 2493.365279597499
n_clusters = 6 --> 2250.4964023486373
n_clusters = 8 --> 2062.22696065942
n_clusters = 10 --> 1831.7764175707493
n_clusters = 20 --> 1347.7699959546637
"""

# plot the clusters and the locations of the centroids at each iteration
for i in range(len(kmeans.centroids_history)):
    centroids_list = kmeans.centroids_history
    labels_list = kmeans.labels_history
    labels = list(set(labels_list[i]))
    kmeans.centroids = centroids_list[i]
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.predict(X))
    # color each centroid
    plt.scatter(centroids_list[i][:, 0], centroids_list[i][:, 1], c=labels, marker='X', edgecolors='black', s=100)
    plt.title(f"Iteration {i}:")
    plt.show()

