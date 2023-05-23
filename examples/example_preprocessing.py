"""
===========================
Applying a fixed pre-processing
===========================

This example shows how custom pre-processing steps can be applied
to the signals by the paradigms in addition to the standard frequency
filtering, cropping and baseline removal.

Additionally, we will see how moving the fixed pre-processing steps
outside of the pipelines can speed up the evaluation process

We will use the P300 paradigm, which uses the AUC as metric.

"""
# Authors: Pierre Guetschel <pierre.guetschel@gmail.com>
#
# License: BSD (3-clause)

import warnings
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import hilbert
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC

import moabb
from moabb.datasets import BNCI2014009
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import P300


##############################################################################
# getting rid of the warnings about the future
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

moabb.set_log_level("info")


##############################################################################
# Defining a transformer
# ----------------------
#
# In this example, we will apply an arbitrary pre-processing that
# first computes the Hilbert transform of the signal,
# then computes its average over a set of pre-defined time intervals.
# In order to be passed to paradigm, the preprocessing-steps have to be implemented as a
# scikit-learn transformer (see https://scikit-learn.org/stable/glossary.html#term-transformer).
# Bellow are two equivalent ways to define such a transformer:
#  1. either as a function wrapped by `sklearn.preprocessing.FunctionTransformer`,
#  2. or as directly as an object with a `transform` method.
#
# The implementations are not the most computationally efficient possible,
# but this will help us demonstrate the speed improvement that can be obtained from
# applying the pre-processing in the paradigm.


def get_average_over_intervals(X: np.array, intervals: List[Tuple[int, int]]):
    X = X.copy()
    X = np.imag(hilbert(X, axis=-1))
    X_list = []
    for i, j in intervals:
        X_list.append(X[:, :, i:j].mean(axis=-1))
    X = np.stack(X_list, axis=-1)
    X = X.reshape(X.shape[0], -1)
    return X


class AverageOverIntervalsTransformer:
    def __init__(self, intervals: List[Tuple[int, int]]):
        self.intervals = intervals

    def fit(self):
        # The eventual fit method of the transformer will not be called by the paradigm.
        # It is not necessary to implement it.
        return self

    def transform(self, X):
        return get_average_over_intervals(X, self.intervals)


sfreq = 256
# the intervals we chose here are arbitrary:
intervals_seconds = [(0.1, 0.2), (0.2, 0.3), (0.3, 0.5)]
intervals = [(int(sfreq * i), int(sfreq * j)) for i, j in intervals_seconds]

# 1. function wrapped by `FunctionTransformer`:
transformer = FunctionTransformer(
    get_average_over_intervals, kw_args=dict(intervals=intervals), validate=False
)
# 2. object with a `transform` method:
transformer2 = AverageOverIntervalsTransformer(intervals=intervals)

##############################################################################
# In the rest of this example, we will use the first implementation (for no particular reason).
#
# Applying the pre-processing to the data
# ---------------------------------------
#
# To apply our pre-processing to the data, we simply have to pass the transformer
# object we defined as an additional argument to the paradigm:

channels = [
    "Fz",
    "FCz",
    "Cz",
    "CPz",
    "Pz",
    "Oz",
    "F3",
    "F4",
    "C3",
    "C4",
    "CP3",
    "CP4",
    "P3",
    "P4",
    "PO7",
    "PO8",
]
paradigm = P300(resample=sfreq, channels=channels, transformer=transformer)
dataset = BNCI2014009()
dataset.subject_list = dataset.subject_list[:3]

X, labels, metadata = paradigm.get_data(dataset, subjects=[1])
assert X.shape[-1] == len(channels) * len(intervals)
print("shape of X:", X.shape)

##############################################################################
# Evaluation
# ----------
#
# Here he will demonstrate the speed improvements that can be obtained from applying the
# fixed pre-processing steps in the paradigm rather than in the pipelines.
#
# For this toy example, we assume that we want to compare two models to classify
# the features obtained from our pre-processing.
# The models are a linear discriminant analysis and a support-vector classifier.
# What we are really interested about is to compare the evaluation speed:
#  - when the pre-processing is done in the paradigm (denoted by the suffix`"_transformer_before"`),
#  - and when the pre-processing is done in the evaluated pipeline (denoted by the suffix`"_transformer_after"`)


pipelines_transformer_before = {}
pipelines_transformer_before["LDA_transformer_before"] = LDA()
pipelines_transformer_before["SVC_transformer_before"] = SVC()

pipelines_transformer_after = {}
pipelines_transformer_after["LDA_transformer_after"] = make_pipeline(transformer, LDA())
pipelines_transformer_after["SVC_transformer_after"] = make_pipeline(transformer, SVC())

overwrite = True  # set to True if we want to overwrite cached results
paradigm_transformer_before = P300(resample=sfreq, transformer=transformer)
paradigm_transformer_after = P300(resample=sfreq)

evaluation_transformer_before = WithinSessionEvaluation(
    paradigm=paradigm_transformer_before,
    datasets=[dataset],
    suffix="_transformer_before",
    overwrite=overwrite,
)
evaluation_transformer_after = WithinSessionEvaluation(
    paradigm=paradigm_transformer_after,
    datasets=[dataset],
    suffix="_transformer_after",
    overwrite=overwrite,
)

results1 = evaluation_transformer_before.process(pipelines_transformer_before)
results2 = evaluation_transformer_after.process(pipelines_transformer_after)
results = pd.concat([results1, results2])

##############################################################################
# Plot Results
# ----------------
#
# Here we plot the results to compare the two pipelines and implementation methods.
#
# First, we check if the two ways of implementing the pipelines are equivalent.

results["classifier"] = results.pipeline.str[:3]
results["transformer_position"] = results.pipeline.str[16:]
results["subject-session"] = results.subject.str.cat(results.session.str[8:], sep="-")

g = sns.catplot(
    data=results,
    kind="point",
    y="score",
    x="transformer_position",
    col="classifier",
    hue="subject-session",
    palette="Set1",
)
g.fig.set_size_inches(8, 4)
g.set(ylabel="ROC AUC", ylim=(0.5, 1))
plt.show()

##############################################################################
# Indeed, we observe that the scores paired by subject and session are almost equal,
# whether the pre-processing is done with the paradigm or with the pipeline.
#
# Now, we will have a look at the computation time in the two cases:

sns.catplot(
    data=results,
    kind="bar",
    y="time",
    x="classifier",
    hue="transformer_position",
    alpha=0.7,
    palette="Set2",
)
g.fig.set_size_inches(8, 4)
g.set(ylabel="computation time [seconds]")
plt.show()

##############################################################################
# We observe a clear advantage for the "before", i.e. when the pre-processing
# is done py the paradigm.
# There are two reasons for the speed gain we observe when the pre-processing is done in the paradigm:
#  1. The pre-processing steps are only computed once for all the pipelines,
#  2. And the pre-processing steps are only computed once for all the cross-validation folds.
