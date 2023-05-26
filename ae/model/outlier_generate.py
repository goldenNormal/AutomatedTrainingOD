from sklearn.mixture import GaussianMixture
import numpy as np
from copulas.multivariate import VineCopula
from copulas.univariate import GaussianKDE
import pandas as pd

seed = 42

def generate_realistic_synthetic(X, y, realistic_synthetic_mode, alpha: int=5, percentage: float=0.1, ratio=0.1):
    '''
    Currently, four types of realistic synthetic outliers can be generated:
    1. local outliers: where normal data follows the GMM distribuion, and anomalies follow the GMM distribution with modified covariance
    2. global outliers: where normal data follows the GMM distribuion, and anomalies follow the uniform distribution
    3. dependency outliers: where normal data follows the vine coupula distribution, and anomalies follow the independent distribution captured by GaussianKDE
    4. cluster outliers: where normal data follows the GMM distribuion, and anomalies follow the GMM distribution with modified mean

    :param X: input X
    :param y: input y
    :param realistic_synthetic_mode: the type of generated outliers
    :param alpha: the scaling parameter for controling the generated local and cluster anomalies
    :param percentage: controling the generated global anomalies
    '''

    if realistic_synthetic_mode in ['local', 'cluster', 'dependency', 'global']:
        pass
    else:
        raise NotImplementedError

    # the number of normal data and anomalies
    pts_n = len(np.where(y == 0)[0])
    # pts_a = len(np.where(y == 1)[0])
    pts_a = int(pts_n * ratio / (1 - ratio))

    # only use the normal data to fit the model
    X = X[y == 0]
    y = y[y == 0]

    # generate the synthetic normal data
    if realistic_synthetic_mode in ['local', 'cluster', 'global']:
        # select the best n_components based on the BIC value
        metric_list = []
        n_components_list = list(np.arange(1, 10))

        for n_components in n_components_list:
            gm = GaussianMixture(n_components=n_components, random_state=seed).fit(X)
            metric_list.append(gm.bic(X))

        best_n_components = n_components_list[np.argmin(metric_list)]

        # refit based on the best n_components
        gm = GaussianMixture(n_components=best_n_components, random_state=seed).fit(X)

        # generate the synthetic normal data
        X_synthetic_normal = gm.sample(pts_n)[0]

    # we found that copula function may occur error in some datasets
    elif realistic_synthetic_mode == 'dependency':
        # sampling the feature since copulas method may spend too long to fit
        if X.shape[1] > 50:
            idx = np.random.choice(np.arange(X.shape[1]), 50, replace=False)
            X = X[:, idx]

        copula = VineCopula('center')  # default is the C-vine copula
        copula.fit(pd.DataFrame(X))

        # sample to generate synthetic normal data
        X_synthetic_normal = copula.sample(pts_n).values

    else:
        pass

    # generate the synthetic abnormal data
    if realistic_synthetic_mode == 'local':
        # generate the synthetic anomalies (local outliers)
        gm.covariances_ = alpha * gm.covariances_
        X_synthetic_anomalies = gm.sample(pts_a)[0]

    elif realistic_synthetic_mode == 'cluster':
        # generate the clustering synthetic anomalies
        gm.means_ = alpha * gm.means_
        X_synthetic_anomalies = gm.sample(pts_a)[0]

    elif realistic_synthetic_mode == 'dependency':
        X_synthetic_anomalies = np.zeros((pts_a, X.shape[1]))

        # using the GuassianKDE for generating independent feature
        for i in range(X.shape[1]):
            kde = GaussianKDE()
            kde.fit(X[:, i])
            X_synthetic_anomalies[:, i] = kde.sample(pts_a)

    elif realistic_synthetic_mode == 'global':
        # generate the synthetic anomalies (global outliers)
        X_synthetic_anomalies = []

        for i in range(X_synthetic_normal.shape[1]):
            low = np.min(X_synthetic_normal[:, i]) * (1 + percentage)
            high = np.max(X_synthetic_normal[:, i]) * (1 + percentage)

            X_synthetic_anomalies.append(np.random.uniform(low=low, high=high, size=pts_a))

        X_synthetic_anomalies = np.array(X_synthetic_anomalies).T

    else:
        pass

    X = np.concatenate((X_synthetic_normal, X_synthetic_anomalies), axis=0)
    y = np.append(np.repeat(0, X_synthetic_normal.shape[0]),
                  np.repeat(1, X_synthetic_anomalies.shape[0]))

    print(X.shape, y.shape)
    return X, y
