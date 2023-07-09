import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB


def show_info(train_df: pd.DataFrame, test_df: pd.DataFrame):
    print('Train set shape:', train_df.shape)
    print('Test set shape:', test_df.shape)
    print(train_df.head())

    print('TRAIN SET MISSING VALUES:')
    print(train_df.isna().sum())
    print('')
    print('TEST SET MISSING VALUES:')
    print(test_df.isna().sum())

    print(f'Duplicates in train set: {train_df.duplicated().sum()}, '
          f'({np.round(100 * train_df.duplicated().sum() / len(train_df), 1)}%)')
    print(f'Duplicates in test set: {test_df.duplicated().sum()}, '
          f'({np.round(100 * test_df.duplicated().sum() / len(test_df), 1)}%)')

    print(train_df.nunique())

    # PLots
    plt.figure(figsize=(6, 6))
    train_df['is_same'].value_counts().plot.pie(explode=[0.1, 0.1],
                                                autopct='%1.1f%%',
                                                shadow=True,
                                                textprops={'fontsize': 16}).set_title("Target distribution")
    plt.figure(figsize=(10, 4))
    sns.histplot(data=train_df, x='euclidean_similarity', hue='is_same', binwidth=1, kde=True)
    plt.title('euclidean similarity distribution')

    plt.figure(figsize=(10, 4))
    sns.histplot(data=train_df, x='cosine_similarity', hue='is_same', binwidth=1, kde=True)
    plt.title('cosine similarity distribution')


def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame, split: bool = True, random_state: int = 69):
    labels = train_df['is_same']
    features = train_df[['class_index1', 'class_index2', 'euclidean_similarity', 'cosine_similarity']]
    #features = train_df[["class_index1_vit" ,"class_index2_vit", "euclidean_similarity_vit", "cosine_similarity_vit" , "class_index1_conv","class_index2_conv", "euclidean_similarity_conv", "cosine_similarity_conv"]]

    features_test = test_df[['class_index1', 'class_index2', 'euclidean_similarity', 'cosine_similarity']]
    #features_test = test_df[["class_index1_vit" ,"class_index2_vit", "euclidean_similarity_vit", "cosine_similarity_vit" , "class_index1_conv","class_index2_conv", "euclidean_similarity_conv", "cosine_similarity_conv"]]

    # Scale numerical data to have mean=0 and variance=1
    numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    features = numerical_transformer.fit_transform(features)
    X_test = numerical_transformer.transform(features_test)

    if split:

        X_train, X_valid, y_train, y_valid = train_test_split(features, labels, stratify=labels, train_size=0.8,
                                                              test_size=0.2, random_state=random_state)

        return X_train, X_valid, y_train, y_valid, X_test

    else:
        return features, labels, X_test


def prepare_grid_search(random_state: int = 69):
    classifiers = {
        "LogisticRegression": LogisticRegression(random_state=random_state),
        "SVC": SVC(random_state=random_state, probability=True),
        "RandomForest": RandomForestClassifier(random_state=random_state),
        "XGBoost": XGBClassifier(random_state=random_state, eval_metric='logloss'),
        "LGBM": LGBMClassifier(random_state=random_state),
        "CatBoost": CatBoostClassifier(random_state=0, verbose=False),
        "NaiveBayes": GaussianNB()

    }

    LR_grid = {'C': [0.25, 0.5, 0.75, 1, 1.25, 1.5],
               'max_iter': [50, 100, 150]}

    SVC_grid = {'C': [0.25, 0.5, 0.75, 1, 1.25, 1.5],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']}

    RF_grid = {'n_estimators': [50, 100, 150, 200, 250, 300],
               'max_depth': [4, 6, 8, 10, 12]}

    boosted_grid = {'n_estimators': [50, 100, 150, 200],
                    'max_depth': [4, 8, 12],
                    'learning_rate': [0.05, 0.1, 0.15]}

    NB_grid = {'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7]}

    # Dictionary of all grids
    grid = {
        "LogisticRegression": LR_grid,
        "SVC": SVC_grid,
        "RandomForest": RF_grid,
        "XGBoost": boosted_grid,
        "LGBM": boosted_grid,
        "CatBoost": boosted_grid,
        "NaiveBayes": NB_grid
    }

    return classifiers, grid


def find_best_classifiers(X_train, X_valid, y_train, y_valid, verbose=True):
    classifiers, grid = prepare_grid_search()

    i = 0
    clf_best_params = classifiers.copy()
    valid_scores = pd.DataFrame({'Classifier': classifiers.keys(),
                                 'Validation accuracy': np.zeros(len(classifiers)),
                                 'Training time': np.zeros(len(classifiers))})
    for key, classifier in classifiers.items():
        start = time.time()
        clf = GridSearchCV(estimator=classifier, param_grid=grid[key], n_jobs=-1, cv=None)

        clf.fit(X_train, y_train)
        valid_scores.iloc[i, 1] = clf.score(X_valid, y_valid)

        clf_best_params[key] = clf.best_params_

        stop = time.time()
        valid_scores.iloc[i, 2] = np.round((stop - start) / 60, 2)

        if verbose:
            print('Model:', key)
            print('Training time (minutes):', valid_scores.iloc[i, 2])
            print('')
        i += 1

    if verbose:
        print(valid_scores)
        print('Best parameters: ', clf_best_params)

    return clf_best_params


def train_classifiers(classifiers, train_df, test_df, folds: int = 5, verbose: bool = True):
    X, y, _ = prepare_data(train_df, test_df, split=False)

    for key, clf in classifiers.items():
        score = 0
        start = time.time()

        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_valid = X[train_idx], X[val_idx]
            y_train, y_valid = y[train_idx], y[val_idx]

            clf.fit(X_train, y_train)
            score += clf.score(X_valid, y_valid)

        stop = time.time()
        score = score / folds

        if verbose:
            print('Model:', key)
            print('Average validation accuracy:', np.round(100 * score, 2))
            print('Training time (minutes):', np.round((stop - start) / 60, 2))

    return classifiers


def predict(classifiers, data):
    predictions = np.zeros(len(data))
    for key, classifier in classifiers.items():
        predictions += classifier.predict_proba(data)[:, 1]

    predictions = predictions / len(classifiers)

    return predictions


def main():
    train = pd.read_csv('../data/train_features(vit).csv')
    test = pd.read_csv('../data/test_features(vit).csv')
    #show_info(train, test)

    X_train, X_valid, y_train, y_valid, X_test = prepare_data(train, test)

    clf_best_params = find_best_classifiers(X_train, X_valid, y_train, y_valid, verbose=True)

    best_classifiers = {
        "RandomForest": RandomForestClassifier(random_state=0),
        "XGBoost": XGBClassifier(**clf_best_params["XGBoost"], random_state=0),
        "LGBM": LGBMClassifier(**clf_best_params["XGBoost"], random_state=0),
        "CatBoost": CatBoostClassifier(**clf_best_params["CatBoost"], verbose=False, random_state=0),
        "LogisticRegression": LogisticRegression(**clf_best_params["LogisticRegression"], random_state=0),
    }

    best_classifiers = train_classifiers(best_classifiers, train, test)

    with open('../assets/classifiers.pickle', 'wb') as file:
        pickle.dump(best_classifiers, file)

    # with open('../assets/classifiers.pickle', 'rb') as file:
    #     best_classifiers = pickle.load(file)
    # best_classifiers = train_classifiers(best_classifiers, train, test)

    valid_predictions = predict(best_classifiers, X_valid) > 0.5

    cm = confusion_matrix(y_valid, valid_predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    #plt.show()
    plt.savefig("../assets/confusion_matrix.png")

    report = classification_report(y_valid, valid_predictions, digits=5)
    #print(report)

    with open('../assets/classification_report.txt', 'w') as file:
        file.write(report)

    dataframe = test.copy()
    X = dataframe.drop(["ID"], axis=1).values


    start = time.time()
    y1 = predict(best_classifiers, X_test)
    finish = time.time()
    print(f"Execution time: {finish - start} seconds")

    y2 = (y1 + 1) % 2

    dataframe["is_same"] = y1
    dataframe["different"] = y2
    dataframe["ID"] = [i for i in range(len(test))]

    result = dataframe[["ID", "is_same", "different"]]

    result['is_same'] = result['is_same'].apply(lambda x: 1 if x > 0.33 else 0)

    result['different'] = (result['is_same'] + 1) % 2

    result['ID'] = [i for i in range(2, len(test)+2)]

    result.to_csv("../data/submission(vit).csv", index=False)


if __name__ == "__main__":
    main()
