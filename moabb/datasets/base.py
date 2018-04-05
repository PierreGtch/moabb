"""
Base class for a dataset
"""
import abc
import logging

log = logging.getLogger()


class BaseDataset(metaclass=abc.ABCMeta):
    """Base dataset"""

    def __init__(self, subjects, sessions_per_subject, events,
                 code, interval, paradigm, task_interval=None, doi=None):
        """
        Parameters required for all datasets

        parameters
        ----------
        subjects: List of int
            List of subject number # TODO: make identifiers more general
        
        sessions_per_subject: int
            Number of sessions per subject

        events: dict of string: int
            String codes for events matched with labels in the stim channel. Currently imagery codes codes can include:
            - left_hand
            - right_hand
            - hands
            - feet
            - rest
            - left_hand_right_foot
            - right_hand_left_foot
            - tongue
            - navigation
            - subtraction
            - word_ass (for word association)

        code: string
            Unique identifier for dataset, used in all plots

        interval: list with 2 entries
            Interval relative to trial start for imagery

        paradigm: ['p300','imagery']
            Defines what sort of dataset this is (currently only imagery is implemented)
        
        task_interval: list of 2 entries or None
            Defines the start and end of the imagery *relative to event marker.* If not specified, defaults to interval. 
        
        doi: DOI for dataset, optional (for now)
        """
        if not isinstance(subjects, list):
            raise(ValueError("subjects must be a list"))

        self.subject_list = subjects
        self.n_sessions = sessions_per_subject
        self.event_id = events
        self.selected_events = events.copy()
        self.code = code
        self.interval = interval
        if task_interval is None:
            assert interval[0]==0, 'Interval does not start at 0 so task onset is necessary'
            self.task_interval = list(interval)
        else:
            if interval[1]-interval[0] > task_interval[1]-task_interval[0]:
                log.warning('Given interval extends outside of imagery period')
            self.task_interval = task_interval
        self.paradigm = paradigm
        self.doi = doi

    def get_data(self, subjects=None):
        """
        Return the data correspoonding to a list of subjects.

        The returned data is a dictionary with the folowing structure::

            data = {'subject_id' :
                        {'session_id':
                            {'run_id': raw}
                        }
                    }

        subjects are on top, then we have sessions, then runs.
        A sessions is a recording done in a single day, without removing the
        EEG cap. A session is constitued of at least one run. A run is a single
        contigous recording. Some dataset break session in multiple runs.

        parameters
        ----------
        subjects: List of int
            List of subject number

        returns
        -------
        data: Dict
            dict containing the raw data
        """
        data = []

        if subjects is None:
            subjects = self.subject_list

        if not isinstance(subjects, list):
            raise(ValueError('subjects must be a list'))

        data = dict()
        for subject in subjects:
            if subject not in self.subject_list:
                raise ValueError('Invalid subject {:d} given'.format(subject))
            data[subject] = self._get_single_subject_data(subject)

        return data

    def download(self, path=None, force_update=False,
                 update_path=None, verbose=None):
        """Download all data from the dataset.

        This function is only usefull to download all the dataset at once.


        Parameters
        ----------
        path : None | str
            Location of where to look for the data storing location.
            If None, the environment variable or config parameter
            ``MNE_DATASETS_(dataset)_PATH`` is used. If it doesn't exist, the
            "~/mne_data" directory is used. If the dataset
            is not found under the given path, the data
            will be automatically downloaded to the specified folder.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            If True, set the MNE_DATASETS_(dataset)_PATH in mne-python
            config to the given path. If None, the user is prompted.
        verbose : bool, str, int, or None
            If not None, override default verbose level
            (see :func:`mne.verbose`).
        """
        for subject in self.subject_list:
            self.data_path(subject=subject, path=path,
                           force_update=force_update,
                           update_path=update_path, verbose=verbose)

    @abc.abstractmethod
    def _get_single_subject_data(self, subject):
        """
        Return the data of a single subject

        The returned data is a dictionary with the folowing structure

        data = {'session_id':
                    {'run_id': raw}
                }

        parameters
        ----------
        subject: int
            subject number

        returns
        -------
        data: Dict
            dict containing the raw data
        """
        pass

    @abc.abstractmethod
    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):
        """Get path to local copy of a subject data.

        Parameters
        ----------
        subject : int
            Number of subject to use
        path : None | str
            Location of where to look for the data storing location.
            If None, the environment variable or config parameter
            ``MNE_DATASETS_(dataset)_PATH`` is used. If it doesn't exist, the
            "~/mne_data" directory is used. If the dataset
            is not found under the given path, the data
            will be automatically downloaded to the specified folder.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            If True, set the MNE_DATASETS_(dataset)_PATH in mne-python
            config to the given path. If None, the user is prompted.
        verbose : bool, str, int, or None
            If not None, override default verbose level
            (see :func:`mne.verbose`).

        Returns
        -------
        path : list of str
            Local path to the given data file. This path is contained inside a
            list of length one, for compatibility.
        """  # noqa: E501
        pass
