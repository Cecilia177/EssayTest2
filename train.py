import numpy as np
from sklearn import svm
from feature import extract_data
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import learning_curve, validation_curve, train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from correlation import pearson_cor, spearman_cor
from sklearn.metrics import make_scorer
from learningcurve import plot_learning_curve
from sklearn.svm import SVR, SVC
from sklearn.feature_selection import RFECV, SelectKBest, f_regression, mutual_info_regression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
import pymysql
import traceback
import math


def cross_val(estimator, params, X_train, y_train, score, cv, n_jobs):
    clf = GridSearchCV(estimator=estimator, param_grid=params, scoring=score, cv=cv, return_train_score=True,
                       n_jobs=n_jobs)
    clf.fit(X_train, y_train)
    return clf.best_estimator_, clf.best_params_


def learning_plot(clf, X_train, y_train, score, cv):
    train_size, train_scores, test_scores, fit_times = learning_curve(
        clf, X_train, y_train, cv=cv, scoring=score,
        train_sizes=[0.1, 0.25, 0.5, 0.75, 1], return_times=True
    )
    train_score_mean = np.mean(train_scores, axis=1)
    test_score_mean = np.mean(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    print(test_score_mean)
    plot(train_size, [train_score_mean, test_score_mean])


def validation_plot(clf, X_train, y_train, score):
    gamma_range = [0.1, 1, 10, 25, 50]
    C_range = [1, 10, 50, 100, 1000]
    # validate parameter gamma
    train_scores_g, test_scores_g = validation_curve(
        clf, X_train, y_train, param_name='gamma', param_range=gamma_range, cv=3, scoring=score)
    # validate parameter C
    train_scores_c, test_scores_c = validation_curve(
        clf, X_train, y_train, param_name='C', param_range=C_range, cv=3, scoring=score
    )
    # get average scores of training and tests
    train_scores_g_mean = np.mean(train_scores_g, axis=1)
    test_scores_g_mean = np.mean(test_scores_g, axis=1)
    scores_list_g = [train_scores_g_mean, test_scores_g_mean]
    plot(gamma_range, scores_list_g)

    train_scores_c_mean = np.mean(train_scores_c, axis=1)
    test_scores_c_mean = np.mean(test_scores_c, axis=1)
    scores_list_c = [train_scores_c_mean, test_scores_c_mean]
    plot(C_range, scores_list_c)


def plot(x_value, y_value_list):
    plt.figure()
    plt.plot(x_value, y_value_list[0], 'o-', color='r', label='Training')
    plt.plot(x_value, y_value_list[1], 'o-', color='g', label='Cross-validation')
    # plt.legend('best')
    plt.show()


def support_machine_regression(X, y, cv):
    parameters = [
        {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]},
        {'kernel': ['rbf'], 'C': [0.1, 0.2, 0.25, 0.35, 0.5, 1, 10, 100, 1000], 'gamma': [0.01, 0.5, 1, 5, 10, 100]},
        # {'kernel': ['poly'], 'C': [1, 10, 100, 1000], 'degree': [3, 4], 'gamma': [0.01, 1, 5, 10, 100]}
    ]
    # define a scoring function
    score_func = make_scorer(pearson_cor, greater_is_better=True)
    svr = SVR()

    # Get the best model through CV
    best_svr, best_params_rg = cross_val(svr, params=parameters, X_train=X,
                                         y_train=y, score=score_func, cv=cv, n_jobs=-1)

    title_rg = r"Learning Curves (SVR, rbf kernel)"
    plt, test_scores = plot_learning_curve(best_svr, title_rg, X, y, ylim=(0.0, 1.0),
                                           cv=cv, n_jobs=4, scoring=score_func)
    plt.show()
    return best_svr, test_scores


def support_machine_classification(X, y, cv):
    parameters = [
        {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]},
        {'kernel': ['rbf'], 'C': [0.1, 0.2, 0.25, 0.35, 0.5, 1, 10, 100, 400, 1000, 2500],
         'gamma': [0.01, 0.5, 1, 5, 10, 100]},
        # {'kernel': ['poly'], 'C': [1, 10, 100, 1000], 'degree': [3, 4], 'gamma': [0.01, 1, 5, 10, 100]}
    ]
    # define a scoring function
    score_func = make_scorer(pearson_cor, greater_is_better=True)
    svc = SVC()
    # Get the best model through CV
    best_svc, best_params_clf = cross_val(svc, params=parameters, X_train=X,
                                          y_train=y, score=score_func, cv=cv, n_jobs=-1)

    title_clf = r"Learning Curves (SVC, rbf kernel)"
    plot_learning_curve(best_svc, title_clf, X, y, ylim=(0.0, 1.0),
                        cv=cv, n_jobs=4, scoring=score_func)
    plt.show()
    print("best svc:", best_svc)
    print("best para:", best_params_clf)
    return best_svc


def mlp_regression(X, y, cv):
    parameters = {
        'alpha': 10.0 ** -np.arange(1, 7)
    }
    score_func = make_scorer(pearson_cor, greater_is_better=True)
    mlp = MLPRegressor(max_iter=800, hidden_layer_sizes=(200, 200), activation='logistic')
    best_mlp, best_params_mlp = cross_val(mlp, params=parameters, X_train=X,
                                          y_train=y, score=score_func, cv=cv, n_jobs=-1)
    title = r"Learning curves (MLP regression)"
    plt, test_scores = plot_learning_curve(best_mlp, title, X, y, ylim=(0.0, 1.0),
                                           cv=cv, n_jobs=4, scoring=score_func)
    plt.show()
    print("best mlp:", best_mlp)
    print("best para:", best_params_mlp)
    return best_mlp, test_scores


def random_forest(X, y, cv):
    score_func = make_scorer(pearson_cor, greater_is_better=True)
    rfg = ExtraTreesRegressor(max_features=8)
    title = r"Learning curves (Random Forest)"
    plt, test_scores = plot_learning_curve(rfg, title, X, y, ylim=(0.0, 1.0),
                                           cv=cv, n_jobs=4, scoring=score_func)
    plt.show()
    return rfg, test_scores


if __name__ == '__main__':
    conn = pymysql.connect(host="127.0.0.1",
                           database='essaydata',
                           port=3306,
                           user='root',
                           password='',
                           charset='utf8')

    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)

    # extract and transform data
    features0 = ['bleu']
    features1 = ['1gram', '2gram', '3gram', '4gram', 'lengthratio']
    features2 = ['1gram', '2gram', '3gram', '4gram', 'lengthratio', 'lsagrade']
    features3 = ['1gram', '2gram', '3gram', '4gram', 'lengthratio', 'vecsim', 'lsagrade', 'fluency']
    features4 = ['bleu', 'lengthratio', 'vecsim', 'lsagrade', 'keymatch', 'np', 'vp']
    features5 = ['keymatch']
    features_all = ['1gram', '2gram', '3gram', '4gram', 'lengthratio', 'lsagrade', 'vecsim', 'fluency', 'np', 'vp',
                    'keymatch']

    # Regression model
    X_rg, y_rg = extract_data(conn, course="201英语一", features=features_all)
    print(len(X_rg))

    # multi-layer perception

    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    X_rg_scaled = scaler.fit_transform(X_rg)



    # features selection
    best_feature_num = -1
    max_score = 0
    best_selection = f_regression
    # for s in [f_regression, mutual_info_regression]:
    #     for n in range(4, 10):
    # selectionKBest = SelectKBest(mutual_info_regression, k=10)
    # selectionKBest.fit(X_rg_scaled, y_rg)
    # print(selectionKBest.scores_)
    # X_rg_selected = selectionKBest.transform(X_rg_scaled)
    # print(X_rg_selected.shape)
    rfr, test_scores = random_forest(X_rg_scaled, y_rg, cv=cv)
    print(rfr)
    print("test scores:", test_scores)

    X_train, X_test, y_train, y_test = train_test_split(X_rg_scaled, y_rg, test_size=0.2, random_state=42)
    rfr.fit(X_train, y_train)
    print(rfr.feature_importances_)
    print("number of features:", rfr.n_features_)
    print("pearson:", pearson_cor(y_test, rfr.predict(X_test)))

    # print("s:", s, "n:", n, "test_scores", test_scores)
    #         if test_scores[-1] > max_score:
    #             best_feature_num = n
    #             max_score = max(max_score, test_scores[-1])
    #             best_selection = s
    # print("best selection:", best_selection)
    # print("selected feature number:", best_feature_num, "max test scores:", max_score)

    # classfication model
    # X_clf, y = extract_data(conn, course="202英语二", features=features3)
    # scaler = MinMaxScaler()
    # X_clf_scaled = scaler.fit_transform(X_clf)
    # y_clf = [str(data) for data in y]
    # svc = classification(X_clf_scaled, y_clf, cv=cv)
    # X_train, X_test, y_train, y_test = train_test_split(X_clf_scaled, y_clf, test_size=0.3, random_state=42)
    # svc.fit(X_train, y_train)
    # y_predict = svc.predict(X_test)
    # print(pearson_cor(y_test, y_predict))
    # print(y_predict)
