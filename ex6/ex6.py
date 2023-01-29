import numpy as np
from matplotlib import pyplot as plt
from ex6_utils import (plot_ims, load_MNIST, outlier_data, gmm_data, plot_2D_gmm, load_dogs_vs_frogs,
                       BayesianLinearRegression, poly_kernel, cluster_purity)
from scipy.special import logsumexp
from typing import Tuple
from scipy.stats import multivariate_normal, norm



def outlier_regression(model: BayesianLinearRegression, X: np.ndarray, y: np.ndarray, p_out: float, T: int,
                       mu_o: float=0, sig_o: float=10) -> Tuple[BayesianLinearRegression, np.ndarray]:
    """
    Gibbs sampling algorithm for robust regression (i.e. regression assuming there are outliers in the data)
    :param model: the Bayesian linear regression that will be used to fit the data
    :param X: the training data, as a numpy array of shape [N, d] where N is the number of points and d is the dimension
    :param y: the regression targets, as a numpy array of shape [N,]
    :param p_out: the assumed probability for outliers in the data
    :param T: number of Gibbs sampling iterations to use in order to fit the model
    :param mu_o: the assumed mean of the outlier points
    :param sig_o: the assumed variance of the outlier points
    :return: the fitted model assuming outliers, as a BayesianLinearRegression model, as well as a numpy array of the
             indices of points which were considered as outliers
    """
    one_minus_p_out = 1 - p_out
    model.fit(X, y, sample=True)
    k_mat = np.full(X.shape, 0)
    k_one = p_out * norm.pdf(y, loc=mu_o, scale=sig_o)
    for i in range(T):
        k_mat = calculate_k_mat(X, k_one, model, one_minus_p_out, y)
        model.fit(X[k_mat == 0], y[k_mat == 0]) #model.fit(X[np.argwhere(k_mat == 0)], y[np.argwhere(k_mat == 0)])
    ind_outliers = np.flatnonzero(k_mat)
    return model, ind_outliers


def calculate_k_mat(X, k_one, model, one_minus_p_out, y):
    logl = model.log_likelihood(X, y)
    k_0 = one_minus_p_out * np.exp(logl)
    k_mat = np.divide(k_one, (k_one + k_0))
    for l in range(len(k_mat)):
        k_mat[l] = np.random.binomial(1, p=k_mat[l])
    return k_mat


def calculate_cov_learn(beta, mu_0, k):
    eye_beta = [beta * np.eye(len(mu_0))]
    eye_beta_k = eye_beta * k
    return np.array(eye_beta_k)


def calculate_log_likelihood(X, k, res, pi, mu, cov):
    for i in range(k):
        pdf_normal = multivariate_normal(mu[i], cov[i]).pdf(X)
        likelihood_k = np.log(pdf_normal) + np.log(pi[i])
        res[:, i] = likelihood_k
    return res


def calculate_log_likelihood_not_learn(X, k, res, pi, mu, mu_0, beta):
    p = np.log(2 * np.pi * beta)
    for i in range(k):
        p_l = np.sum((X - mu[i]) ** 2, axis=1)
        res[:, i] = - (((len(mu_0) * p) + (p_l / beta)) / 2) + np.log(pi[i])
    return res

class BayesianGMM:
    def __init__(self, k: int, alpha: float, mu_0: np.ndarray, sig_0: float, nu: float, beta: float,
                 learn_cov: bool=True):
        """
        Initialize a Bayesian GMM model
        :param k: the number of clusters to use
        :param alpha: the value of alpha to use for the Dirichlet prior over the mixture probabilities
        :param mu_0: the mean of the prior over the means of each Gaussian in the GMM
        :param sig_0: the variance of the prior over the means of each Gaussian in the GMM
        :param nu: the nu parameter of the inverse-Wishart distribution used as a prior for the Gaussian covariances
        :param beta: the variance of the inverse-Wishart distribution used as a prior for the covariances
        :param learn_cov: a boolean indicating whether the cluster covariances should be learned or not
        """
        self.k = k
        self.mu_0 = mu_0
        self.sig_0 = sig_0
        self.nu = nu
        self.alpha = alpha
        self.beta = beta
        self.learn_cov = learn_cov
        self.eye_mu_0 = np.eye(len(mu_0))
        self.mu = np.random.multivariate_normal(mu_0, self.eye_mu_0 * sig_0, size=k)
        self.pi = np.ones(k) / k
        if learn_cov:
            self.cov = calculate_cov_learn(beta, mu_0, k)
        if not learn_cov:
            self.mu_numer = np.divide(self.mu_0, self.sig_0)

    def log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the log-likelihood of each data point under each Gaussian in the GMM
        :param X: the data points whose log-likelihood should be calculated, as a numpy array of shape [N, d]
        :return: the log-likelihood of each point under each Gaussian in the GMM
        """
        res = np.zeros((len(X), self.k))
        if self.learn_cov:
            res = calculate_log_likelihood(X, self.k, res, self.pi, self.mu, self.cov)
        if not self.learn_cov:
            res = calculate_log_likelihood_not_learn(X, self.k, res, self.pi, self.mu, self.mu_0, self.beta)
        return res

    def cluster(self, X: np.ndarray) -> np.ndarray:
        """
        Clusters the data according to the learned GMM
        :param X: the data points to be clustered, as a numpy array of shape [N, d]
        :return: a numpy array containing the indices of the Gaussians the data points are most likely to belong to,
                 as a numpy array with shape [N,]
        """
        log_like = self.log_likelihood(X)
        result = np.argmax(log_like, axis=1)
        return result

    def calculate_z(self):
        x_l = self.log_likelihood(self.X)
        z = np.zeros(len(x_l))
        for i, log_likelihoods in enumerate(x_l):
            probabilities = np.exp(log_likelihoods - logsumexp(log_likelihoods))
            z[i] = np.random.choice(range(self.k), 1, p=probabilities)[0]
        return z

    def calculate_pi(self, z):
        alpha_mat = self.alpha * np.ones(self.k)
        for i in range(len(z)):
            index = int(z[i])
            alpha_mat[index] += 1
        self.pi = np.random.dirichlet(alpha_mat)

    def calculate_mu_cov(self, z):
        if self.learn_cov:
            self.calculate_mu_cov_learn(z)

        else:  # no learn cov, Fixed covariance
            self.calculate_cov_mean_not_learn(z)

    def calculate_cov_mean_not_learn(self, z):
        self.mu = np.random.multivariate_normal(self.mu_0, self.sig_0 * self.eye_mu_0, size=self.k)
        for i in np.unique(z).astype(int):
            d = (np.count_nonzero(z == i) / self.beta) + (1 / self.sig_0)
            mu = ((np.sum(self.X[z == i, :], axis=0) / self.beta) + self.mu_numer) / d
            cov = self.eye_mu_0 / d
            self.mu[i] = np.random.multivariate_normal(mu, cov)

    def calculate_mu_cov_learn(self, z):
        for i in range(self.k):
            inv_cov_k = np.linalg.inv(np.linalg.cholesky(self.cov[i]))
            inv_cov_k = inv_cov_k.T @ inv_cov_k

            mu_k_cov_1 = np.count_nonzero(z == i) * inv_cov_k + (1 / self.sig_0) * np.eye(len(self.cov[i]))

            inv_mu_k_cov = np.linalg.inv(np.linalg.cholesky(mu_k_cov_1))
            inv_mu_k_cov = inv_mu_k_cov.T @ inv_mu_k_cov

            mu_k_mu = inv_cov_k @ np.sum(self.X[z == i, :], axis=0) + (self.mu_0 / self.sig_0)
            mu_k_mu = np.linalg.solve(mu_k_cov_1, mu_k_mu)

            self.mu[i] = np.random.multivariate_normal(mu_k_mu, inv_mu_k_cov)

            self.update_cov(i, z)

    def update_cov(self, i, z):
        x_k_minus_mu_s = self.X[z == i, :] - self.mu[i]
        self.cov[i] = (self.nu * self.beta * self.eye_mu_0 +
                       x_k_minus_mu_s.T @ x_k_minus_mu_s) / (self.nu
                                                             + np.count_nonzero(z == i))

    def gibbs_fit(self, X: np.ndarray, T: int) -> 'BayesianGMM':
        """
        Fits the Bayesian GMM model using a Gibbs sampling algorithm
        :param X: the training data, as a numpy array of shape [N, d] where N is the number of points
        :param T: the number of sampling iterations to run the algorithm
        :return: the fitted model
        """
        self.X = X
        for i in range(T):
            print(i)
            z = self.calculate_z()
            self.calculate_pi(z)
            self.calculate_mu_cov(z)
        return self

if __name__ == '__main__':
    # ------------------------------------------------------ section 2 - Robust Regression
    # ---------------------- question 2
    # load the outlier data
    x, y = outlier_data(50)
    # init BLR model that will be used to fit the data
    mdl = BayesianLinearRegression(theta_mean=np.zeros(2), theta_cov=np.eye(2), sample_noise=0.15)

    # sample using the Gibbs sampling algorithm and plot the results
    plt.figure()
    plt.scatter(x, y, 15, 'k', alpha=.75)
    xx = np.linspace(-0.2, 5.2, 100)
    for t in [0, 1, 5, 10, 25]:
        samp, outliers = outlier_regression(mdl, x, y, T=t, p_out=0.1, mu_o=4, sig_o=2)
        plt.plot(xx, samp.predict(xx), lw=2, label=f'T={t}')
    plt.xlim([np.min(xx), np.max(xx)])
    plt.legend()
    plt.show()

    # ---------------------- question 3
    # load the images to use for classification
    N = 1000
    ims, labs = load_dogs_vs_frogs(N)
    # define BLR model that should be used to fit the data
    mdl = BayesianLinearRegression(sample_noise=0.001, kernel_function=poly_kernel(2))
    # use Gibbs sampling to sample model and outliers
    samp, outliers = outlier_regression(mdl, ims, labs, p_out=0.01, T=50, mu_o=0, sig_o=.5)
    # plot the outliers
    plot_ims(ims[outliers], title='outliers')
    #
    # ------------------------------------------------------ section 3 - Bayesian GMM
    # ---------------------- question 5
    # load 2D GMM data
    k, N = 5, 1000
    X = gmm_data(N, k)

    for i in range(5):
        gmm = BayesianGMM(k=50, alpha=.01, mu_0=np.zeros(2), sig_0=.5, nu=5, beta=.5)
        gmm.gibbs_fit(X, T=100)

        # plot a histogram of the mixture probabilities (in descending order)
        pi = gmm.pi  # mixture probabilities from the fitted GMM
        plt.figure()
        plt.bar(np.arange(len(pi)), np.sort(pi)[::-1])
        plt.ylabel(r'$\pi_k$')
        plt.xlabel('cluster number')
        plt.show()

        # plot the fitted 2D GMM
        plot_2D_gmm(X, gmm.mu, np.array(gmm.cov), gmm.cluster(X))  # the second input are the means and the third are the covariances

    # ---------------------- questions 6-7
    # load image data
    MNIST, labs = load_MNIST()
    # flatten the images
    ims = MNIST.copy().reshape(MNIST.shape[0], -1)
    gmm = BayesianGMM(k=500, alpha=1, mu_0=0.5*np.ones(ims.shape[1]), sig_0=.1, nu=1, beta=.25, learn_cov=False)
    gmm.gibbs_fit(ims, 100)

    # plot a histogram of the mixture probabilities (in descending order)
    pi = gmm.pi  # mixture probabilities from the fitted GMM
    plt.figure()
    plt.bar(np.arange(len(pi)), np.sort(pi)[::-1])
    plt.ylabel(r'$\pi_k$')
    plt.xlabel('cluster number')
    plt.show()

    # find the clustering of the images to different Gaussians
    cl = gmm.cluster(ims)
    clusters = np.unique(cl)
    print(f'{len(clusters)} clusters used')
    # calculate the purity of each of the clusters
    purities = np.array([cluster_purity(labs[cl == k]) for k in clusters])
    purity_inds = np.argsort(purities)

    # plot 25 images from each of the clusters with the top 5 purities
    for ind in purity_inds[-5:]:
        clust = clusters[ind]
        plot_ims(MNIST[cl == clust][:25].astype(float), f'cluster {clust}: purity={purities[ind]:.2f}')

    # plot 25 images from each of the clusters with the bottom 5 purities
    for ind in purity_inds[:5]:
        clust = clusters[ind]
        plot_ims(MNIST[cl == clust][:25].astype(float), f'cluster {clust}: purity={purities[ind]:.2f}')

