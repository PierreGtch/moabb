import logging
from abc import ABCMeta, abstractmethod, abstractproperty
from collections import OrderedDict
from pathlib import Path
import pickle
import json

import mne
import numpy as np
import pandas as pd

from moabb.datasets import download as dl
from moabb.analysis.results import get_digest

log = logging.getLogger(__name__)


class BaseParadigm(metaclass=ABCMeta):
    """Base Paradigm."""

    def __init__(self):
        pass

    @property
    def param_names(self):
        """
        This property lists the parameter names of the Paradigm,
        i.e. in theory, the arguments of __init__ method.
        This property should be updated in subclasses if new arguments are added.
        """
        return []

    def get_params(self):
        """
        Get parameters for this estimator.
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self.param_names:
            value = getattr(self, key)
            out[key] = value
        return out

    def __repr__(self):
        params = self.get_params()
        # sort the parameters by name :
        params = OrderedDict((k, v) for k, v in sorted(params.items()))
        return f"{self.__class__.__name__}({', '.join(repr(k)+'='+repr(v) for k, v in params.items())})"

    @abstractproperty
    def scoring(self):
        """Property that defines scoring metric (e.g. ROC-AUC or accuracy
        or f-score), given as a sklearn-compatible string or a compatible
        sklearn scorer.

        """
        pass

    @abstractproperty
    def datasets(self):
        """Property that define the list of compatible datasets"""
        pass

    @abstractmethod
    def is_valid(self, dataset):
        """Verify the dataset is compatible with the paradigm.

        This method is called to verify dataset is compatible with the
        paradigm.

        This method should raise an error if the dataset is not compatible
        with the paradigm. This is for example the case if the
        dataset is an ERP dataset for motor imagery paradigm, or if the
        dataset does not contain any of the required events.

        Parameters
        ----------
        dataset : dataset instance
            The dataset to verify.
        """
        pass

    def prepare_process(self, dataset):
        """Prepare processing of raw files

        This function allows to set parameter of the paradigm class prior to
        the preprocessing (process_raw). Does nothing by default and could be
        overloaded if needed.

        Parameters
        ----------

        dataset : dataset instance
            The dataset corresponding to the raw file. mainly use to access
            dataset specific information.
        """
        pass

    def _epochs_to_array(self, epochs, dataset):
        # rescale to work with uV
        X = dataset.unit_factor * epochs.get_data()
        k = len(self.filters)
        if k > 1:
            # if more than one band, return a 4D
            if X.shape[0] % k != 0:
                raise ValueError('Invalid number of epochs')
            # XXX previously, using return_epochs=True and multiple bands was strangely implemented,
            # XXX the epochs order was N*k*n
            # XXX with k the number of bands, n the number of epochs per runs and N the total number of runs
            # XXX it makes more sense to have N*n*k
            k_n, c, t = X.shape
            n = k_n // k
            X = X.reshape(n, k, c, t).transpose((0, 2, 3, 1))
            # otherwise return a 3D array
        return X

    def process_raw(self, raw, dataset, return_epochs=False):  # noqa: C901
        """
        Process one raw data file.

        This function apply the preprocessing and eventual epoching on the
        individual run, and return the data, labels and a dataframe with
        metadata.

        metadata is a dataframe with as many row as the length of the data
        and labels.

        Parameters
        ----------
        raw: mne.Raw instance
            the raw EEG data.
        dataset : dataset instance
            The dataset corresponding to the raw file. mainly use to access
            dataset specific information.
        return_epochs: boolean
            This flag specifies whether to return only the data array or the
            complete processed mne.Epochs

        returns
        -------
        X : Union[np.ndarray, mne.Epochs]
            the data that will be used as features for the model
            Note: if return_epochs=True,  this is mne.Epochs
            if return_epochs=False, this is np.ndarray
        labels: np.ndarray
            the labels for training / evaluating the model
        metadata: pd.DataFrame
            A dataframe containing the metadata

        """
        epochs, labels, metadata = self.process_raw_to_epochs(raw, dataset)
        if return_epochs:
            return epochs, labels, metadata
        return self._epochs_to_array(epochs), labels, metadata

    def process_raw_to_epochs(self, raw, dataset):  # noqa: C901
        """
        Process one raw data file.

        This function apply the preprocessing and eventual epoching on the
        individual run, and return the data, labels and a dataframe with
        metadata.

        metadata is a dataframe with as many row as the length of the data
        and labels.

        Parameters
        ----------
        raw: mne.Raw instance
            the raw EEG data.
        dataset : dataset instance
            The dataset corresponding to the raw file. mainly use to access
            dataset specific information.

        returns
        -------
        X : mne.Epochs
            the data that will be used as features for the model
        labels: np.ndarray
            the labels for training / evaluating the model
        metadata: pd.DataFrame
            A dataframe containing the metadata

        """
        # get events id
        event_id = self.used_events(dataset)

        # find the events, first check stim_channels then annotations
        stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)
        if len(stim_channels) > 0:
            events = mne.find_events(raw, shortest_event=0, verbose=False)
        else:
            try:
                events, _ = mne.events_from_annotations(
                    raw, event_id=event_id, verbose=False
                )
            except ValueError:
                log.warning("No matching annotations in {}".format(raw.filenames))
                return

        # picks channels
        if self.channels is None:
            picks = mne.pick_types(raw.info, eeg=True, stim=False)
        else:
            picks = mne.pick_channels(
                raw.info["ch_names"], include=self.channels, ordered=True
            )

        # pick events, based on event_id
        try:
            events = mne.pick_events(events, include=list(event_id.values()))
        except RuntimeError:
            # skip raw if no event found
            return

        # get interval
        tmin = self.tmin + dataset.interval[0]
        if self.tmax is None:
            tmax = dataset.interval[1]
        else:
            tmax = self.tmax + dataset.interval[0]

        X = []
        for bandpass in self.filters:
            fmin, fmax = bandpass
            # filter data
            raw_f = raw.copy().filter(
                fmin, fmax, method="iir", picks=picks, verbose=False
            )
            # epoch data
            baseline = self.baseline
            if baseline is not None:
                baseline = (
                    self.baseline[0] + dataset.interval[0],
                    self.baseline[1] + dataset.interval[0],
                )
                bmin = baseline[0] if baseline[0] < tmin else tmin
                bmax = baseline[1] if baseline[1] > tmax else tmax
            else:
                bmin = tmin
                bmax = tmax
            epochs = mne.Epochs(
                raw_f,
                events,
                event_id=event_id,
                tmin=bmin,
                tmax=bmax,
                proj=False,
                baseline=baseline,
                preload=True,
                verbose=False,
                picks=picks,
                event_repeated="drop",
                on_missing="ignore",
            )
            if bmin < tmin or bmax > tmax:
                epochs.crop(tmin=tmin, tmax=tmax)
            if self.resample is not None:
                epochs = epochs.resample(self.resample)
            X.append(epochs)

        inv_events = {k: v for v, k in event_id.items()}
        labels = np.array([inv_events[e] for e in epochs.events[:, -1]])
        # here we change the order of the epochs so that multiple epochs objects can be easily concatenated
        k = len(X) # number of bands
        n = len(X[0]) # number of epochs in the run
        X = mne.concatenate_epochs([X[k_i][n_i] for n_i in range(n) for k_i in range(k)])
        metadata = pd.DataFrame(index=range(len(labels)))
        return X, labels, metadata

    def _get_preprocessed_sign(self, dataset):
        return "PREPROCESSED_" + dataset.__class__.__name__.upper() + "_" + get_digest(self)

    def preprocessed_path(self, dataset, path=None):
        sign = self._get_preprocessed_sign(dataset)
        path = dl.get_dataset_path(sign, path)
        key_dest = "MNE-{:s}".format(sign.lower())
        return str(Path(path) / key_dest)

    def _find_preprocessed_data(self, dataset, subject):
        path = self.preprocessed_path(dataset)
        desc_file = Path(path) / str(subject) / "desc.json"
        if not desc_file.is_file():
            # if the data of this subject has not been pre-processed:
            return None
        with open(desc_file, "r") as f:
            sessions = json.load(f)
        return sessions

    def load_processed_epochs(self, dataset, subject, session, run, preload=False):
        """
        TODO
        """
        path = self.preprocessed_path(dataset)
        dir = (Path(path) / str(subject) / str(session) / str(run)).expanduser().resolve()
        epochs = mne.read_epochs(str(dir/'epochs-epo.fif'), preload=preload)
        with dir.joinpath("labels.pickle").open("rb") as f:
            labels = pickle.load(f)
        with dir.joinpath("metadata.pickle").open("rb") as f:
            metadata = pickle.load(f)
        return epochs, labels, metadata

    def _save_preprocessed_epochs(self, proc, dataset, subject, session, run):
        path = Path(self.preprocessed_path(dataset))
        if not path.is_dir():
            # create the main save directory if it does not exist
            path.mkdir(parents=True)
            # save the paradigm name and parameters
            desc = dict(
                paradigm_name=self.__class__.__name__,
                dataset_name=dataset.__class__.__name__,
                paradigm_params=self.get_params())
            with path.joinpath("desc.json").open("w") as f:
                json.dump(desc, f)
        if proc is None:
            return
        epochs, labels, metadata = proc
        dir = (path / str(subject) / str(session) / str(run)).expanduser()
        if not dir.is_dir():
            # create the run's directory if it does not exist
            dir.mkdir(parents=True)
        epochs.save(str(dir/'epochs-epo.fif'))
        with dir.joinpath("labels.pickle").open("wb") as f:
            pickle.dump(labels, f)
        with dir.joinpath("metadata.pickle").open("wb") as f:
            pickle.dump(metadata, f)

    def _mark_as_saved(self, dataset, subject, sessions):
        path = Path(self.preprocessed_path(dataset))
        desc_file = Path(path) / str(subject) / "desc.json"
        if not desc_file.parent.is_dir():
            desc_file.parent.mkdir(parents=True)
        sessions = {session : {run : None for run in runs.keys()} for session,runs in sessions.items()}
        with open(desc_file, "w") as f:
            json.dump(sessions, f)


    def get_data(self, dataset, subjects=None, return_epochs=False, preload=False, use_preprocessed=True, save_preprocessed=True):
        """
        Return the data for a list of subject.

        return the data, labels and a dataframe with metadata. the dataframe
        will contain at least the following columns

        - subject : the subject indice
        - session : the session indice
        - run : the run indice

        parameters
        ----------
        dataset:
            A dataset instance.
        subjects: List of int
            List of subject number
        return_epochs: boolean
            This flag specifies whether to return only the data array or the
            complete processed mne.Epochs
        preload: boolean
            This flag specifies whether the epochs must be pre-loaded in memory.
            Only applied if you are using saved processed data and if return_epochs is True.
        use_preprocessed: XXX
        save_preprocessed: XXX

        returns
        -------
        X : Union[np.ndarray, mne.Epochs]
            the data that will be used as features for the model
            Note: if return_epochs=True,  this is mne.Epochs
            if return_epochs=False, this is np.ndarray
        labels: np.ndarray
            the labels for training / evaluating the model
        metadata: pd.DataFrame
            A dataframe containing the metadata.
        """
        if not self.is_valid(dataset):
            message = "Dataset {} is not valid for paradigm".format(dataset.code)
            raise AssertionError(message)

        self.prepare_process(dataset)

        X = []
        labels = []
        metadata = []
        for subject in subjects:
            sessions = self._find_preprocessed_data(dataset, subject) if use_preprocessed else None
            is_preprocessed = True
            if sessions is None:
                is_preprocessed = False
                sessions = dataset._get_single_subject_data(subject)
            for session, runs in sessions.items():
                for run, raw_or_None in runs.items():
                    if is_preprocessed:
                        proc = self.load_processed_epochs(dataset, subject, session, run, preload=preload)
                    else:
                        proc = self.process_raw_to_epochs(raw_or_None, dataset)
                        if save_preprocessed:
                            self._save_preprocessed_epochs(proc, dataset, subject, session, run)

                    if proc is None:
                        # this mean the run did not contain any selected event
                        # go to next
                        continue

                    x, lbs, met = proc
                    met["subject"] = subject
                    met["session"] = session
                    met["run"] = run
                    metadata.append(met)

                    X.append(x)
                    labels = np.append(labels, lbs, axis=0)
            if save_preprocessed and not is_preprocessed:
                self._mark_as_saved(dataset, subject, sessions)
        metadata = pd.concat(metadata, ignore_index=True)
        X = mne.concatenate_epochs(X)
        if not return_epochs:
            X = self._epochs_to_array(X, dataset)
        return X, labels, metadata
