import pandas as pd
from sources.utils.url_utils import create_filenames_csv
from sources.utils.add_features import add_features
from sources.predict import predict


def create_submission(dataframe: pd.DataFrame) -> pd.DataFrame:
    # dataframe = create_filenames_csv(dataframe)
    # dataframe = add_features(dataframe)

    dataframe = dataframe.copy()
    X = dataframe.drop(["ID"], axis=1).values
    y = predict(X)

    dataframe["is_same"] = y
    dataframe["ID"] = [i for i in range(2, 22662)]

    submission = dataframe[["ID", "is_same"]]

    return submission


def main():
    test = pd.read_csv("../data/results.csv")
    submission = create_submission(test)
    submission.to_csv("../data/submission.csv", index=False)


if __name__ == "__main__":
    main()
