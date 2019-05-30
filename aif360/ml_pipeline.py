import numpy as np
from sklearn.preprocessing import StandardScaler

from aif360.metrics import ClassificationMetric
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing

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
        self.dataset_train = []
        self.dataset_test = []
        self.test_predictions = []


    def fit(self, dataset, test_frac=0.3, threshold=0.5, feature_scaling=False):
        """
        Trains our model on the dataset.

        Args:
            dataset (aif360.datasets.StructuredDataset): An instance of a structured dataset
            test_frac (float): A real number between 0 and 1 denoting the % of the dataset to be used as test data
            threshold (float): A real number between 0 and 1 denoting the threshold of acceptable class imbalance
        """

        if test_frac < 0 or test_frac > 1:
            raise ValueError("Parameter test_frac must be between 0 and 1")

        dataset_train, dataset_test = dataset.split([1-test_frac], shuffle=False)

        # If a preprocessing function was supplied, transform the data first
        if self.preprocessor:
            dataset_train = self.preprocessor.fit_transform(dataset_train)
            dataset_test = self.preprocessor.fit_transform(dataset_test)

        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        
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

        self.test_predictions = dataset_test_pred

    
    def evaluate(self, metric):
        """
        Evaluates an AIF360 metric against the trained model

        Args:
            metric (aif360.metrics.Metric): An AIF360 metric class
        Raises:
            AttributeError: If a model has not been trained, an AttributeError will be raised
        Returns:
            float: A float denoting the performance of the specified metric on the trained model
        """

        import re
        
        if not self.dataset_train:
            raise AttributeError("A model must be fit before evaluating a metric")

        # Try instantiating the metric using only the dataset argument. If it's the classification metric, then we'll fall through to the
        # except statement since that metric requires an extra argument.
        try:
            curr_metric = metric(self.dataset_test, unprivileged_groups=self.unprivileged, privileged_groups=self.privileged)
        except TypeError:
            curr_metric = metric(self.dataset_test, self.test_predictions, unprivileged_groups=self.unprivileged, privileged_groups=self.privileged)
        
        # TODO: Figure out a way to swtich this out for other methods of evaluation
        return curr_metric.mean_difference()


if __name__ == '__main__':
    from datasets import AdultDataset
    from metrics import BinaryLabelDatasetMetric, ClassificationMetric
    from sklearn.linear_model import LogisticRegression
    
    # Example. Load the adult dataset, train with LogisticRegression and postprocess with EqOddsPostProcessing 
    DATASET = AdultDataset()
    PRIVILIGED_GROUPS = [{'race': 1}] # white = 1
    UNPRIVILIGED_GROUPS = [{'race': 0}]
    METRICS = [ClassificationMetric, BinaryLabelDatasetMetric]

    # Instantiate our pre/postprocessing algorithms
    postprocessor = CalibratedEqOddsPostprocessing(privileged_groups=PRIVILIGED_GROUPS, unprivileged_groups=UNPRIVILIGED_GROUPS)
    
    # Create our pipeline object and fit our model
    mlp = MLPipeline(LogisticRegression(), PRIVILIGED_GROUPS, UNPRIVILIGED_GROUPS, postprocessor=postprocessor)
    mlp.fit(DATASET)
    
    # Evaluate our metrics
    for metric in METRICS:
        mlp.evaluate(metric)



