import numpy as np
from matplotlib import pyplot as plt
from typing import Callable


def polynomial_basis_functions(degree: int) -> Callable:
    """
    Create a function that calculates the polynomial basis functions up to (and including) a degree
    :param degree: the maximal degree of the polynomial basis functions
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             polynomial basis functions, a numpy array of shape [N, degree+1]
    """
    def pbf(x: np.ndarray):
        H = np.zeros(shape=(len(x), degree + 1))
        for d in range(degree + 1):
            H[:, d] = np.power(x.T, d)
        return H
    return pbf


def gaussian_basis_functions(centers: np.ndarray, beta: float) -> Callable:
    """
    Create a function that calculates Gaussian basis functions around a set of centers
    :param centers: an array of centers used by the basis functions
    :param beta: a float depicting the length scale of the Gaussians
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             Gaussian basis functions, a numpy array of shape [N, len(centers)+1]
    """
    def gbf(x: np.ndarray):
        H = np.zeros(shape=(len(x), len(centers) + 1))
        H[:, 0] = np.ones(len(x))
        for i in range(1, len(centers) + 1):
            H[:, i] = np.exp(-(x - centers[i - 1]) ** 2/ (2 * (beta ** 2)))
        return H
    return gbf


def spline_basis_functions(knots: np.ndarray) -> Callable:
    """
    Create a function that calculates the cubic regression spline basis functions around a set of knots
    :param knots: an array of knots that should be used by the spline
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             cubic regression spline basis functions, a numpy array of shape [N, len(knots)+4]
    """
    def csbf(x: np.ndarray):
        H = np.zeros(shape=(len(x), len(knots) + 4))
        for d in range(4):
            H[:, d] = np.power(x.T, d)
        for i in range(4, len(knots) + 4):
            cur = (np.power(x - knots[i - 4], 3))
            cur[cur < 0] = 0
            H[:, i] = cur
        return H
    return csbf


def learn_prior(hours: np.ndarray, temps: np.ndarray, basis_func: Callable) -> tuple:
    """
    Learn a Gaussian prior using historic data
    :param hours: an array of vectors to be used as the 'X' data
    :param temps: a matrix of average daily temperatures in November, as loaded from 'jerus_daytemps.npy', with shape
                  [# years, # hours]
    :param basis_func: a function that returns the design matrix of the basis functions to be used
    :return: the mean and covariance of the learned covariance - the mean is an array with length dim while the
             covariance is a matrix with shape [dim, dim], where dim is the number of basis functions used
    """
    thetas = []
    # iterate over all past years
    for i, t in enumerate(temps):
        ln = LinearRegression(basis_func).fit(hours, t)
        thetas.append(ln.theta)  # append learned parameters here

    thetas = np.array(thetas)

    # take mean over parameters learned each year for the mean of the prior
    mu = np.mean(thetas, axis=0)
    # calculate empirical covariance over parameters learned each year for the covariance of the prior
    cov = (thetas - mu[None, :]).T @ (thetas - mu[None, :]) / thetas.shape[0]
    return mu, cov


class BayesianLinearRegression:
    def __init__(self, theta_mean: np.ndarray, theta_cov: np.ndarray, sig: float, basis_functions: Callable):
        """
        Initializes a Bayesian linear regression model
        :param theta_mean:          the mean of the prior
        :param theta_cov:           the covariance of the prior
        :param sig:                 the signal noise to use when fitting the model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        """
        self.theta_mean = theta_mean
        self.theta_cov = theta_cov
        self.sig = sig
        self.basis_functions = basis_functions
        self.posterior_cov = None
        self.posterior_mean = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianLinearRegression':
        """
        Find the model's posterior using the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        H = self.basis_functions(X)
        inv_theta_cov = np.linalg.inv(np.linalg.cholesky(self.theta_cov))
        inv_theta_cov = inv_theta_cov.T @ inv_theta_cov
        inv_sig = 1 / self.sig
        C = np.linalg.inv(np.linalg.cholesky(inv_theta_cov + inv_sig * (H.T @ H)))
        self.posterior_cov = C.T @ C
        self.posterior_mean = self.posterior_cov @ (H.T * inv_sig @ y + inv_theta_cov @ self.theta_mean)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model using MMSE
        :param X: the samples to predict
        :return: the predictions for X
        """
        return self.basis_functions(X) @ self.posterior_mean

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Find the model's posterior and return the predicted values for X using MMSE
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the model's standard deviation around the mean prediction for the values of X
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the standard deviations (same shape as X)
        """
        H = self.basis_functions(X)
        return np.sqrt(np.diagonal(H @ self.posterior_cov @ H.T) + self.sig)

    def posterior_sample(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model and sampling from the posterior
        :param X: the samples to predict
        :return: the predictions for X
        """
        return self.basis_functions(X) @ np.random.multivariate_normal(self.posterior_mean, self.posterior_cov)


class LinearRegression:
    def __init__(self, basis_functions: Callable):
        """
        Initializes a linear regression model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        """
        self.basis_funcs = basis_functions
        self.theta = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the model to the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        H = self.basis_funcs(X)
        self.theta = np.linalg.pinv(H) @ y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model
        :param X: the samples to predict
        :return: the predictions for X
        """
        return self.basis_funcs(X) @ self.theta

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit the model and return the predicted values for X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)


def plot_linear_regression(ln, train, train_hours, test, test_hours, title=""):
    plt.scatter(test_hours, test, color="g", label='train')
    plt.scatter(train_hours, train, color="r", label='test')
    y_pred = ln.predict(test_hours)
    plt.plot(test_hours, y_pred, color="b", label='predicted')
    plt.legend(loc='best')
    plt.xlabel('hour')
    plt.ylabel('temp')
    plt.title(title)
    plt.show()


def plot_prior(x, mu, cov, sig, pbf, title=""):
    H = pbf(x)
    mean = H @ mu
    std = np.sqrt(np.diagonal(H @ cov @ H.T) + sig)
    plt.fill_between(x, mean - std, mean + std, alpha=.5, label='confidence interval')
    plt.plot(x, mean, label='mean')
    for i in range(5):
        mean_sample = np.random.multivariate_normal(mu, cov)
        y_sample = H @ mean_sample
        plt.plot(x, y_sample)
    plt.title(title)
    plt.xlabel('hour')
    plt.ylabel('temperature')
    plt.legend()
    plt.figure()
    plt.show()


def plot_posterior(blr, test_hours, test, basis_func , title=""):
    H = basis_func(test_hours)
    mean = H @ blr.posterior_mean
    std = blr.predict_std(test_hours)
    plt.fill_between(test_hours, mean - std, mean + std, alpha=.5, label='confidence interval')
    plt.plot(test_hours, mean, label='MMSE')
    plt.scatter(test_hours, test, label='Test')
    for i in range(5):
        plt.plot(test_hours, blr.posterior_sample(test_hours))
    plt.title(title)
    plt.xlabel('hour')
    plt.ylabel('temperature')
    plt.legend()
    plt.figure()
    plt.show()


def main():
    # load the data for November 16 2020
    nov16 = np.load('nov162020.npy').astype(float)
    nov16_hours = np.arange(0, 24, .5)
    train = nov16[:len(nov16)//2]
    train_hours = nov16_hours[:len(nov16)//2]
    test = nov16[len(nov16)//2:]
    test_hours = nov16_hours[len(nov16)//2:]

    # setup the model parameters
    degrees = [3, 7]
    # ----------------------------------------- Classical Linear Regression
    for d in degrees:
        ln = LinearRegression(polynomial_basis_functions(d)).fit(train_hours, train)

        # print average squared error performance
        print(f'Average squared error with LR and d={d} is {np.mean((test - ln.predict(test_hours)) ** 2):.2f}')

        # plot graphs for linear regression part
        title = f"Classical linear regression with d = {d}"
        plot_linear_regression(ln, train, train_hours, nov16, nov16_hours, title=title)

    # ----------------------------------------- Bayesian Linear Regression

    # load the historic data
    temps = np.load('jerus_daytemps.npy').astype(np.float64)
    hours = np.array([2, 5, 8, 11, 14, 17, 20, 23]).astype(np.float64)
    x = np.arange(0, 24, .1)

    # setup the model parameters
    sigma = 0.25
    degrees = [3, 7]  # polynomial basis functions degrees
    beta = 3  # length scale for Gaussian basis functions

    # sets of centers S_1, S_2, and S_3
    centers = [np.array([6, 12, 18]),
               np.array([4, 8, 12, 16, 20]),
               np.array([2, 4, 8, 12, 16, 20, 22])]

    # sets of knots K_1, K_2 and K_3 for the regression splines
    knots = [np.array([12]),
             np.array([8, 16]),
             np.array([6, 12, 18])]

    # ---------------------- polynomial basis functions
    for deg in degrees:
        pbf = polynomial_basis_functions(deg)
        mu, cov = learn_prior(hours, temps, pbf)
        blr = BayesianLinearRegression(mu, cov, sigma, pbf)
        blr.fit(train_hours, train)

        # plot prior graphs
        title = f"Mean function by prior\npolynomial basic functions degree: {deg}"
        plot_prior(x, mu, cov, sigma, pbf, title)
        # plot posterior graphs
        plot_posterior(blr, test_hours, test, pbf, title)
        print(f'Average squared error with BlR and d={deg} is {np.mean((test - blr.predict(test_hours)) ** 2):.2f}')

    # # ---------------------- Gaussian basis functions
    for ind, c in enumerate(centers):
        rbf = gaussian_basis_functions(c, beta)
        mu, cov = learn_prior(hours, temps, rbf)

        blr = BayesianLinearRegression(mu, cov, sigma, rbf)
        blr.fit(train_hours, train)
        title = ""
        # plot prior graphs
        title = f"Mean function by prior\nGaussian basis functions with {c} centers"

        plot_prior(x, mu, cov, sigma, rbf, title)
        title = f"Mean function by posterior,\nGaussian basis functions with {c} centers"
        # plot posterior graphs
        plot_posterior(blr, test_hours, test, rbf, title)
        # print average squared error performance
        print(f'Average squared error of BlR with gaussian basic functions with {c} centers: {np.mean((test - blr.predict(test_hours)) ** 2):.2f}')
    # # ---------------------- cubic regression splines
    for ind, k in enumerate(knots):
        spline = spline_basis_functions(k)
        mu, cov = learn_prior(hours, temps, spline)

        blr = BayesianLinearRegression(mu, cov, sigma, spline)
        blr.fit(train_hours, train)

        # plot prior graphs
        title = f"Mean function by prior\nCubic regression with {k} knots"
        plot_prior(x, mu, cov, sigma, spline, title)

        # plot posterior graphs
        title = f"Mean function by posterior,\nCubic regression with {k} knots"
        plot_posterior(blr, test_hours, test, spline, title)

        print(f'Average squared error of BlR with Cubic regression with {k} knots'
              f' {np.mean((test - blr.predict(test_hours)) ** 2):.2f}')
if __name__ == '__main__':
    main()
