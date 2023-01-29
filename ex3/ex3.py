import numpy as np
from matplotlib import pyplot as plt
from ex3_utils import BayesianLinearRegression, polynomial_basis_functions, load_prior


def log_evidence(model: BayesianLinearRegression, X, y):
    """
    Calculate the log-evidence of some data under a given Bayesian linear regression model
    :param model: the BLR model whose evidence should be calculated
    :param X: the observed x values
    :param y: the observed responses (y values)
    :return: the log-evidence of the model on the observed data
    """
    # extract the variables of the prior distribution
    mu = model.mu
    sig = model.cov
    n = model.sig

    # extract the variables of the posterior distribution
    model.fit(X, y)
    map = model.fit_mu
    map_cov = model.fit_cov

    # calculate the log-evidence
    H = model.h(X)
    det_cov_prior = np.linalg.det(sig)
    det_cov_posterior = np.linalg.det(map_cov)

    log_posterior_prior = (1/2) * np.log(det_cov_posterior/det_cov_prior)

    inv_prior_cov = np.linalg.inv(np.linalg.cholesky(sig))
    inv_prior_cov = inv_prior_cov.T @ inv_prior_cov

    first = (map - mu).T @ inv_prior_cov @ (map - mu)
    second = (1/n) * (np.linalg.norm(y - H @ map) ** 2)
    third = len(X) * np.log(n)

    middle = -(1/2) * (first + second + third)
    end = 2 * np.pi
    return log_posterior_prior + middle - end


def main():
    # ------------------------------------------------------ section 2.1
    # set up the response functions
    f1 = lambda x: x**2 - 1
    f2 = lambda x: -x**4 + 3*x**2 + 50*np.sin(x/6)
    f3 = lambda x: .5*x**6 - .75*x**4 + 2.75*x**2
    f4 = lambda x: 5 / (1 + np.exp(-4*x)) - (x - 2 > 0)*x
    f5 = lambda x: np.cos(x*4) + 4*np.abs(x - 2)
    functions = [f1, f2, f3, f4, f5]
    x = np.linspace(-3, 3, 500)

    # set up model parameters
    degrees = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    noise_var = .25
    alpha = 5

    # go over each response function and polynomial basis function
    for i, f in enumerate(functions):
        y = f(x) + np.sqrt(noise_var) * np.random.randn(len(x))
        ev_list = []
        models_list = []
        for j, d in enumerate(degrees):
            # set up model parameters
            pbf = polynomial_basis_functions(d)
            mean, cov = np.zeros(d + 1), np.eye(d + 1) * alpha
            # calculate evidence
            ev = log_evidence(BayesianLinearRegression(mean, cov, noise_var, pbf), x, y)
            ev_list.append(ev)
            models_list.append(BayesianLinearRegression(mean, cov, noise_var, pbf))
        best_index = ev_list.index(min(ev_list))
        worst_index = ev_list.index(max(ev_list))

        # plot evidence versus degree and predicted fit
        plt.figure()
        plt.plot(degrees, ev_list)
        plt.xlabel('degree')
        plt.ylabel("log evidence")
        plt.title(f"f{i + 1} evidence vs degree")
        plt.savefig(f'f{i + 1} evidence vs degree.png')
        plt.show()
        plt.close()

        plt.figure()
        plt.plot(x, y, 'og', label="y points", markersize=3)
        worst_model = models_list[worst_index].fit(x, y)
        pred_worst = worst_model.predict(x)
        std_worst = worst_model.predict_std(x)
        plt.plot(x, pred_worst, label=f"worst fit, degree {degrees[worst_index]}")
        diff_std = pred_worst - std_worst
        sum_std = pred_worst + std_worst
        plt.fill_between(x, diff_std, sum_std, alpha=.5, label="worst confidence")

        bst_m = models_list[best_index].fit(x, y)
        pred_best = bst_m.predict(x)
        std_best = bst_m.predict_std(x)
        plt.plot(x, pred_best, label=f"best fit, degree {degrees[best_index]}")
        diff_std_b = pred_worst - std_worst
        sum_std_b = pred_worst + std_worst
        plt.fill_between(x, diff_std_b, sum_std_b, alpha=.5, label="best confidence")


        plt.xlabel('x')
        plt.ylabel(f'f={i + 1}')
        plt.title(f"f={i + 1} best and worst model")
        plt.legend()
        plt.savefig(f'f{i + 1} best_worst.png')
        plt.show()
        plt.close()

    # ------------------------------------------------------ section 2.2
    # load relevant data
    nov16 = np.load('nov162020.npy')
    hours = np.arange(0, 24, .5)
    train = nov16[:len(nov16) // 2]
    hours_train = hours[:len(nov16) // 2]

    # load prior parameters and set up basis functions
    mu, cov = load_prior()
    pbf = polynomial_basis_functions(7)

    noise_vars = np.linspace(.05, 2, 100)
    evs = np.zeros(noise_vars.shape)
    highest_ev = -10000
    ev_noise = 0
    for i, n in enumerate(noise_vars):
        # calculate the evidence
        mdl = BayesianLinearRegression(mu, cov, n, pbf)
        ev = log_evidence(mdl, hours_train, train)
        evs[i] = ev
        if highest_ev < ev:
            highest_ev = ev
            ev_noise = n
    # plot log-evidence versus amount of sample noise
    plt.figure()
    plt.plot(noise_vars, evs)
    plt.xlabel('sigma^2')
    plt.ylabel("log evidence")
    plt.title(f"log evidence score as function of $f='sigma^2' the highest ev: {round(highest_ev, 2)} with noise {round(ev_noise, 2)}")
    plt.savefig("log evidence vs noise")
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()



