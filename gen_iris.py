import pandas as pd
import mlrun
from sklearn.datasets import load_iris


def iris_generator(context, format="csv"):
    """a function which generates the iris dataset"""
    iris = load_iris()
    iris_dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_labels = pd.DataFrame(data=iris.target, columns=["label"])
    iris_dataset = pd.concat([iris_dataset, iris_labels], axis=1)

    context.logger.info("saving iris dataframe")
    context.log_result("label_column", "label")
    context.log_dataset("dataset", df=iris_dataset, format=format, index=False)


if __name__ == "__main__":
    with mlrun.get_or_create_ctx("iris_generator", upload_artifacts=True) as context:
        iris_generator(context, context.get_param("format", "csv"))
