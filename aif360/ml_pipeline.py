import numpy as np
from sklearn.preprocessing import StandardScaler

from metrics import ClassificationMetric
from algorithms.postprocessing import CalibratedEqOddsPostprocessing

class MLPipeline(object):

    """
    Defines a machine-learning pipeline for evaluating fairness in predictors. For usage, see example at the bottom of the file.

    Args:
        model (sklearn.model): An sklearn predictor
        privileged (list[dict[str, float]]): A list of dictionaries with keys representing privileged attribute + value pairs
        unprivileged (list[dict[str, float]]): A list of dictionaries with keys representing unprivileged attribute + value pairs
        preprocessor (aif360.algorithms.preprocessing): An instance of an AIF360 preprocessing algorithm
        postprocessor (aif360.algorithms.postprocessing): An instance of an AIF360 postprocessing algorithm
    """

    def __init__(self, model, privileged=[], unprivileged=[], preprocessor=None, postprocessor=None):
        self.model = model
        self.privileged = privileged
        self.unprivileged = unprivileged
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor


    def fit(self, dataset, n=0.3, threshold=0.5, feature_scaling=False, metrics=[]):
        """
        Trains our model on the dataset.

        Args:
            dataset (aif360.datasets.StructuredDataset): An instance of a structured dataset
            n (float): A real number between 0 and 1 denoting the % of the dataset to be used as test data
            threshold (float): A real number between 0 and 1 denoting the threshold of acceptable class imbalance
            metrics (list[DatasetMetric]): A list of AIF360 metric class objects
        """

        if n < 0 or n > 1:
            raise ValueError("Parameter n must be between 0 and 1")

        dataset_train, dataset_test = dataset.split([1-n], shuffle=False)

        # If a preprocessing function was supplied, transform the data first
        if self.preprocessor:
            dataset_train = self.preprocessor.fit_transform(dataset_train)
            dataset_test = self.preprocessor.fit_transform(dataset_test)
        
        X_train = dataset_train.features
        y_train = dataset_train.labels.ravel()

        X_test = dataset_test.features
        y_test = dataset_test.labels.ravel()

        # Apply feature scaling if specified
        if feature_scaling:
            scaler = StandardScaler().fit(dataset_train.features)
            X_train = scaler.transform(dataset_train.features)
            X_test = scaler.transform(dataset_test.features)
        
        # Train our model
        self.model.fit(X_train, y_train)

        fav_idx = np.where(self.model.classes_ == dataset_train.favorable_label)[0][0]
        y_test_pred_prob = self.model.predict_proba(X_test)[:, fav_idx]
        dataset_test_pred = dataset_test.copy(deepcopy=True)
        
        y_test_pred = np.zeros_like(dataset_test.labels)
        y_test_pred[y_test_pred_prob >= threshold] = dataset_test.favorable_label
        y_test_pred[~(y_test_pred_prob >= threshold)] = dataset_test.unfavorable_label
        dataset_test_pred.labels = y_test_pred

        # If a postprocessor was specified, transform the test results
        if self.postprocessor:
            dataset_test_pred = self.postprocessor.fit(dataset_test, dataset_test_pred) \
                                                  .predict(dataset_test_pred)

        # Evaluate our metrics
        for i, metric in enumerate(metrics):

            # Try instantiating the metric using only the dataset argument. If it's the classification metric, then we'll fall through to the
            # except statement since that metric requires an extra argument.
            try:
                curr_metric = metric(dataset_test, unprivileged_groups=self.unprivileged, privileged_groups=self.privileged)

                # TODO: Change this to something user-specified
                print("METRIC ({})".format(i+1))
                print("========================")
                print(curr_metric.mean_difference(), "\n")
            except TypeError:
                curr_metric = metric(dataset_test, dataset_test_pred, unprivileged_groups=self.unprivileged, privileged_groups=self.privileged)

                # TODO: Change this to something user-specified
                print("METRIC ({})".format(i+1))
                print("========================")
                print(curr_metric.mean_difference(), "\n")
                


if __name__ == '__main__':
    from datasets import AdultDataset
    from metrics import BinaryLabelDatasetMetric, ClassificationMetric
    from sklearn.linear_model import LogisticRegression
    
    # Example. Load the adult dataset, train with LogisticRegression and postprocess with EqOddsPostProcessing 
    dataset = AdultDataset()
    privileged_groups = [{'race': 1}] # white = 1
    unprivileged_groups = [{'race': 0}]
    postprocessor = CalibratedEqOddsPostprocessing(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
    
    # Create our pipeline object, fit and evaluate our metrics.
    # TODO: Perhaps the metric evaluation could be moved outside of the fit function.
    mlp = MLPipeline(LogisticRegression(), privileged_groups, unprivileged_groups, postprocessor=postprocessor)
    mlp.fit(dataset, metrics=[ClassificationMetric, BinaryLabelDatasetMetric])



