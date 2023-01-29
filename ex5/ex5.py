import numpy as np
from matplotlib import pyplot as plt
from ex5_utils import load_im_data, GaussianProcess, RBF_kernel, accuracy, Gaussian, plot_ims


def prediction_function(input_data, model1, model2):
    log_likelihood1 = model1.log_likelihood(input_data)
    log_likelihood2 = model2.log_likelihood(input_data)
    likelihood_difference = np.clip(log_likelihood1 - log_likelihood2, -25, 25)
    return likelihood_difference


def calculate_posterior_mean(prior_mean, prior_variance, sample_variance, data, sample_size):
    numerator = (1/sample_variance)*np.sum(data, axis = 0) + (1/prior_variance)*prior_mean
    denominator = sample_size*(1/sample_variance) + (1/prior_variance)
    return numerator/denominator


def plot_class_separator_estimate(mean1, mean2, feature_set):
    print(feature_set.shape)
    mean_diff = mean1 - mean2
    numerator = np.linalg.norm(mean1)**2 - np.linalg.norm(mean2)**2
    y = numerator/(2*mean_diff[1]) - (mean_diff[0]/mean_diff[1])*feature_set
    print(feature_set.shape)
    print(y.shape)
    plt.plot(feature_set[:0], y[:0], color="green")
    plt.plot(feature_set[:, 1], y[:, 1], color="green", label="Estimated boundary")

def plot_class_separator(mean1, mean2, feature_set):
    mean_diff = mean1 - mean2
    numerator = np.linalg.norm(mean1)**2 - np.linalg.norm(mean2)**2
    y = numerator/(2*mean_diff[1]) - (mean_diff[0]/mean_diff[1])*feature_set
    plt.plot(feature_set, y, color='gray')


def plot_data(x_p, x_m, title):
    plt.figure()
    plt.scatter(x_p[:, 0], x_p[:, 1], 10, label="positive")
    plt.scatter(x_m[:, 0], x_m[:, 1], 10, label=r"negative")
    plt.title(title)
    plt.legend(loc='upper left', fontsize=5)
    plt.show()


def plot_data2(combined_data, mu_m, mu_p, updated_negative_mean, updated_positive_mean, x_m, x_p):
    fig, ax = plt.subplots()
    ax.scatter(x_p[:, 0], x_p[:, 1], s=20, c='purple', marker='o', label='Positive')
    ax.scatter(x_m[:, 0], x_m[:, 1], s=20, c='orange', marker='x', label='Negative')
    ax.scatter(mu_p[0], mu_p[1], s=100, c='black', marker='P', label='Prior Positive Mean')
    ax.scatter(mu_m[0], mu_m[1], s=100, c='black', marker='X', label='Prior Negative Mean')
    ax.scatter(updated_positive_mean[0], updated_positive_mean[1], s=100, c='blue', marker='P',
               label='Posterior Positive Mean')
    ax.scatter(updated_negative_mean[0], updated_negative_mean[1], s=100, c='blue', marker='X',
               label='Posterior Negative Mean')
    ax.set_title("Decision boundary")
    plot_class_separator_estimate(updated_positive_mean, updated_negative_mean, combined_data)
    ax.legend(loc='upper left', fontsize=12)
    plt.show()

def main():
    # ------------------------------------------------------ section 1
    # define question variables
    sig, sig_0 = 0.1, 0.25
    mu_p, mu_m = np.array([1, 1]), np.array([-1, -1])

    # sample 5 points from each class
    np.random.seed(0)
    x_p = np.array([.5, 0])[None, :] + np.sqrt(sig) * np.random.randn(5, 2)
    x_m = np.array([-.5, -.5])[None, :] + np.sqrt(sig) * np.random.randn(5, 2)

    combined_data = np.concatenate([x_p, x_m])
    plot_data(x_p, x_m, "Sample of 5 points positive and 5 negative")
    print(combined_data.shape)
    updated_positive_mean = calculate_posterior_mean(mu_p, sig_0, sig, x_p, x_p.shape[0])
    updated_negative_mean = calculate_posterior_mean(mu_m, sig_0, sig, x_m, x_m.shape[0])

    plot_data2(combined_data, mu_m, mu_p, updated_negative_mean, updated_positive_mean, x_m, x_p)

    post_covariance = (1 / (x_m.shape[0] * (1 / sig) + 1 / sig_0)) * np.eye(x_m.shape[1])

    for i in range(10):
        sample_positive_mean = np.random.multivariate_normal(updated_positive_mean, post_covariance)
        sample_negative_mean = np.random.multivariate_normal(updated_negative_mean, post_covariance)
        plot_class_separator(sample_positive_mean, sample_negative_mean, combined_data)

    plot_class_separator_estimate(updated_positive_mean, updated_negative_mean, combined_data)
    plt.title("10 Decision Boundaries by sampling the class means from the posteriors")
    plt.legend(loc='upper left', fontsize=7)
    plt.savefig("Q1 10 Decision Boundaries")
    plt.show()


    # ------------------------------------------------------ section 2
    # load image data
    (dogs, dogs_t), (frogs, frogs_t) = load_im_data()

    # split into train and test sets
    train = np.concatenate([dogs, frogs], axis=0)
    labels = np.concatenate([np.ones(dogs.shape[0]), -np.ones(frogs.shape[0])])
    test = np.concatenate([dogs_t, frogs_t], axis=0)
    labels_t = np.concatenate([np.ones(dogs_t.shape[0]), -np.ones(frogs_t.shape[0])])

    # ------------------------------------------------------ section 2.1
    nus = [0, 1, 5, 10, 25, 50, 75, 100]
    train_score, test_score = np.zeros(len(nus)), np.zeros(len(nus))
    for i, nu in enumerate(nus):
        beta = .05 * nu
        print(f'QDA with nu={nu}', end='', flush=True)
        g1 = Gaussian(beta, nu).fit(dogs)
        g2 = Gaussian(beta, nu).fit(frogs)

        y_train = prediction_function(train, g1, g2)
        y_test = prediction_function(test, g1, g2)

        train_score[i] = accuracy(y_train, labels)
        test_score[i] = accuracy(y_test, labels_t)

        print(f': train={train_score[i]:.2f}, test={test_score[i]:.2f}', flush=True)

    plt.figure()
    plt.plot(nus, train_score, lw=2, label='train')
    plt.plot(nus, test_score, lw=2, label='test')
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel(r'value of $\nu$')
    plt.show()

    # ------------------------------------------------------ section 2.2
    # define question variables
    kern, sigma = RBF_kernel(.009), .1
    Ns = [250, 500, 1000, 3000, 5750]
    train_score, test_score = np.zeros(len(Ns)), np.zeros(len(Ns))

    gp = None
    for i, N in enumerate(Ns):
        print(f'GP using {N} samples', end='', flush=True)

        dogs_train = np.random.choice(dogs.shape[0], N, replace=False)
        frogs_train = dogs_train + dogs.shape[0]
        train_indices = np.concatenate([dogs_train,frogs_train])
        cur_X_train = train[train_indices]
        cur_y_train = labels[train_indices]

        gp = GaussianProcess(kern, sigma).fit(cur_X_train, cur_y_train)
        y_train = gp.predict(train)
        y_test = gp.predict(test)

        train_score[i] = accuracy(y_train, labels)
        test_score[i] = accuracy(y_test, labels_t)

        print(f': train={train_score[i]:.2f}, test={test_score[i]:.2f}', flush=True)

    plt.figure()
    plt.plot(Ns, train_score, lw=2, label='train')
    plt.plot(Ns, test_score, lw=2, label='test')
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('# of samples')
    plt.xscale('log')
    plt.show()

    # calculate how certain the model is about the predictions
    d = np.abs(gp.predict(dogs_t) / gp.predict_std(dogs_t))
    inds = np.argsort(d)
    # plot most and least confident points
    plot_ims(dogs_t[inds][:25], 'least confident')
    plot_ims(dogs_t[inds][-25:], 'most confident')


if __name__ == '__main__':
    main()







