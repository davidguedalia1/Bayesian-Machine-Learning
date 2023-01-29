import numpy as np
from typing import Callable
from matplotlib import pyplot as plt

KERNEL_STRS = {
    'Laplacian': r'Laplacian, $\alpha={}$, $\beta={}$',
    'RBF': r'RBF, $\alpha={}$, $\beta={}$',
    'Gibbs': r'Gibbs, $\alpha={}$, $\beta={}$, $\delta={}$, $\gamma={}$',
    'NN': r'NN, $\alpha={}$, $\beta={}$'
}


def average_error(pred: np.ndarray, vals: np.ndarray):
    """
    Calculates the average squared error of the given predictions
    :param pred: the predicted values
    :param vals: the true values
    :return: the average squared error between the predictions and the true values
    """
    return np.mean((pred - vals)**2)


def RBF_kernel(alpha: float, beta: float) -> Callable:
    """
    An implementation of the RBF kernel
    :param alpha: the kernel variance
    :param beta: the kernel bandwidth
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """
    def kern(x, y):
        exp_cal = -beta * np.sum((x - y) ** 2)
        return alpha * np.exp(exp_cal)
    return kern


def Laplacian_kernel(alpha: float, beta: float) -> Callable:
    """
    An implementation of the Laplacian kernel
    :param alpha: the kernel variance
    :param beta: the kernel bandwidth
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """
    def kern(x, y):
        exp_cal = -beta * np.sum(np.abs(x - y), axis=-1)
        return alpha * np.exp(exp_cal)
    return kern


def Gibbs_kernel(alpha: float, beta: float, delta: float, gamma: float) -> Callable:
    """
    An implementation of the Gibbs kernel (see section 4.2.3 of http://www.gaussianprocess.org/gpml/chapters/RW.pdf)
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """
    def kern(x, y):
        l_x = alpha * np.exp(-beta * np.sum((x - delta) ** 2)) + gamma
        l_y = alpha * np.exp(-beta * np.sum((y - delta) ** 2)) + gamma
        sum_squres = (l_x ** 2) + (l_y ** 2)
        root = np.sqrt((2 * l_x * l_y) / sum_squres)
        return root * np.exp(-1 * np.sum((x - y) ** 2) / sum_squres)
    return kern


def NN_kernel(alpha: float, beta: float) -> Callable:
    """
    An implementation of the Neural Network kernel (see section 4.2.3 of http://www.gaussianprocess.org/gpml/chapters/RW.pdf)
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """
    def kern(x, y):
        numerator = 2 * beta * (np.dot(x, y) + 1)
        denominator = np.sqrt((1 + 2 * beta * (1 + np.dot(x, x))) * (1 + 2 * beta * (1 + np.dot(y, y))))
        return (alpha * (2 / np.pi)) * np.arcsin(numerator / denominator)
    return kern


class GaussianProcess:

    def __init__(self, kernel: Callable, noise: float):
        """
        Initialize a GP model with the specified kernel and noise
        :param kernel: the kernel to use when fitting the data/predicting
        :param noise: the sample noise assumed to be added to the data
        """
        self.noise = noise
        self.kernel = kernel
        self.posterior_mean = None
        self.posterior_cov = None
        self.X = None
        self.C_n = None
        self.N = None
        self.K_X = None
        self.alpha = None

    def fit(self, X, y) -> 'GaussianProcess':
        """
        Find the model's posterior using the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        self.X = X
        self.N = len(y)
        C = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                C[i, j] = self.kernel(self.X[i : i + 1], self.X[j : j + 1])
        noise = self.noise * np.eye(self.N)
        self.C_n = C + noise
        K_noise = np.linalg.inv(np.linalg.cholesky(self.C_n))
        self.cov_inv = K_noise.T @ K_noise
        self.alpha = self.cov_inv @ y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the MMSE regression values of X with the trained model
        :param X: the samples to predict
        :return: the predictions for X
        """
        predict_N = len(X)
        self.K_X = np.empty((self.N, predict_N))
        for i in range(self.N):
            for j in range(predict_N):
                self.K_X[i, j] = self.kernel(self.X[i], X[j])
        pred = self.alpha @ self.K_X
        self.posterior_mean = pred
        return pred

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Find the model's posterior and return the predicted values for X using MMSE
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)

    def sample(self, X) -> np.ndarray:
        """
        Sample a function from the posterior
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the sample (same shape as X)
        """
        return np.random.multivariate_normal(self.posterior_mean, self.posterior_cov)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the model's standard deviation around the mean prediction for the values of X
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the standard deviations (same shape as X)
        """
        C = np.zeros((len(X), len(X)))
        for i in range(len(X)):
            for j in range(len(X)):
                C[i, j] = self.kernel(X[i : i + 1], X[j : j + 1])
        self.posterior_cov = C - (self.K_X.T @ np.linalg.solve(self.C_n, self.K_X))
        sqrt_poster = np.sqrt(np.diagonal(self.posterior_cov))
        return sqrt_poster

    def log_evidence(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates the model's log-evidence under the training data
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the log-evidence of the model under the data points
        """
        self.fit(X, y)
        y_p = np.dot(y, self.alpha)
        det = np.log(np.linalg.det(np.eye(X.shape[0]) * self.noise))
        return -(1/2) * (y_p + det + self.N * np.log(2 * np.pi))

    def calculate_gram_mat(self, X):
        N = len(X)
        k_mat = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                k_mat[i, j] = self.kernel(X[i], X[j])
        k_mat = k_mat + k_mat.T
        for i in range(N):
            k_mat[i, i] = self.kernel(X[i], X[i])
        return k_mat


def main():
    # ------------------------------------------------------ section 2.1
    xx = np.linspace(-5, 5, 500)
    x, y = np.array([-2, -1, 0, 1, 2]), np.array([-2.1, -4.3, 0.7, 1.2, 3.9])

    # ------------------------------ questions 2 and 3
    # choose kernel parameters
    params = [
        # Laplacian kernels
        ['Laplacian', Laplacian_kernel, 1, 0.25],           # insert your parameters, order: alpha, beta
        ['Laplacian', Laplacian_kernel, 1, 0.5],        # insert your parameters, order: alpha, beta
        ['Laplacian', Laplacian_kernel, 2, 0.75],        # insert your parameters, order: alpha, beta

        # RBF kernels
        ['RBF', RBF_kernel, 1, 0.25],                       # insert your parameters, order: alpha, beta
        ['RBF', RBF_kernel, 2, 0.5],                    # insert your parameters, order: alpha, beta
        ['RBF', RBF_kernel, 2, 0.75],                    # insert your parameters, order: alpha, beta

        # Gibbs kernels
        ['Gibbs', Gibbs_kernel, 1, 0.5, 0, .1],             # insert your parameters, order: alpha, beta, delta, gamma
        ['Gibbs', Gibbs_kernel, 2, 0.5, 0, 0.5],    # insert your parameters, order: alpha, beta, delta, gamma
        ['Gibbs', Gibbs_kernel, 0.5, 0.75, 1, 0.25],    # insert your parameters, order: alpha, beta, delta, gamma

        # Neurel network kernels
        ['NN', NN_kernel, 1, 0.25],                         # insert your parameters, order: alpha, beta
        ['NN', NN_kernel, 1, 2],                      # insert your parameters, order: alpha, beta
        ['NN', NN_kernel, 2, 0.25],                      # insert your parameters, order: alpha, beta
    ]
    noise = 0.05

    # plot all of the chosen parameter settings
    for p in params:
        # create kernel according to parameters chosen
        k = p[1](*p[2:])    # p[1] is the kernel function while p[2:] are the kernel parameters

        # initialize GP with kernel defined above
        gp = GaussianProcess(k, noise)

        # plot prior variance and samples from the priors
        plt.figure()
        gram_mat = gp.calculate_gram_mat(xx)
        std = np.sqrt(np.diagonal(gram_mat))
        n = np.zeros(len(xx))

        plt.plot(xx, n, label="Mean Prior")
        plt.fill_between(xx, n - std, n + std, alpha=.4, label="CI")

        for i in range(5):
            prior = np.random.multivariate_normal(n, gram_mat)
            plt.plot(xx, prior)

        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.title(KERNEL_STRS[p[0]].format(*p[2:]))
        plt.ylim([-5, 5])

        # fit the GP to the data and calculate the posterior mean and confidence interval
        gp.fit(x, y)
        m, s = gp.predict(xx), 2*gp.predict_std(xx)

        # plot posterior mean, confidence intervals and samples from the posterior
        plt.figure()
        plt.fill_between(xx, m-s, m+s, alpha=.3)
        plt.plot(xx, m, lw=2)
        for i in range(6): plt.plot(xx, gp.sample(xx), lw=1)
        plt.scatter(x, y, 30, 'k')
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.title(KERNEL_STRS[p[0]].format(*p[2:]))
        plt.ylim([-5, 5])
        plt.show()

    # ------------------------------ question 4
    # define range of betas
    betas = np.linspace(.1, 15, 101)
    noise = .15

    # calculate the evidence for each of the kernels
    evidence = [GaussianProcess(RBF_kernel(1, beta=b), noise).log_evidence(x, y) for b in betas]

    # plot the evidence as a function of beta
    plt.figure()
    plt.plot(betas, evidence, lw=2)
    plt.xlabel(r'$\beta$')
    plt.ylabel('log-evidence')
    plt.show()

    # extract betas that had the min, median and max evidence
    srt = np.argsort(evidence)
    min_ev, median_ev, max_ev = betas[srt[0]], betas[srt[(len(evidence)+1)//2]], betas[srt[-1]]

    # plot the mean of the posterior of a GP using the extracted betas on top of the data
    plt.figure()
    plt.scatter(x, y, 30, 'k', alpha=.5)
    plt.plot(xx, GaussianProcess(RBF_kernel(1, beta=min_ev), noise).fit(x, y).predict(xx), lw=2, label='min evidence')
    plt.plot(xx, GaussianProcess(RBF_kernel(1, beta=median_ev), noise).fit(x, y).predict(xx), lw=2, label='median evidence')
    plt.plot(xx, GaussianProcess(RBF_kernel(1, beta=max_ev), noise).fit(x, y).predict(xx), lw=2, label='max evidence')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.legend()
    plt.show()

    # ------------------------------------------------------ section 2.2
    # define function and parameters
    f = lambda x: np.sin(x*3)/2 - np.abs(.75*x) + 1
    xx = np.linspace(-3, 3, 100)
    noise = .25
    beta = 2

    # calculate the function values
    np.random.seed(0)
    y = f(xx) + np.sqrt(noise)*np.random.randn(len(xx))

    # ------------------------------ question 5
    # fit a GP model to the data
    gp = GaussianProcess(kernel=RBF_kernel(1, beta=beta), noise=noise).fit(xx, y)

    # calculate posterior mean and confidence interval
    m, s = gp.predict(xx), 2*gp.predict_std(xx)
    print(f'Average squared error of the GP is: {average_error(m, y):.2f}')

    # plot the GP prediction and the data
    plt.figure()
    plt.fill_between(xx, m-s, m+s, alpha=.5)
    plt.plot(xx, m, lw=2)
    plt.scatter(xx, y, 30, 'k', alpha=.5)
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim([-3, 3])
    plt.show()


if __name__ == '__main__':
    main()



