# All helper functions are written in this file.
import numpy as np
from feature_eng import FeatureEng
import pdb
import json
import time
from json import JSONEncoder
from joblib import Parallel, delayed


def IO():
    """
        This is wraper function around MNE input/oupt modules
    """
    from mne import io

    return io


def eeg_reader(data_path, labels, orders, epoch_length=1e4):
    """
    This is custome writen function to read eegdata from .mat file which saved in h5 format.
    It has some preassumptions about data structure (3 channel data)
    It also takes care of signal synchronicity with labels

    INPUTS
    data_path: full path to location of data (.mat file)
    labels: labels corresponding to data
    orders: channel order
    epoch_length: length of epoch for each label
    z_score: zscore function need to be applied to signal or not
    """

    # import packages
    import numpy as np
    import h5py
    from scipy.stats import zscore

    # reading files from h5 source
    F = h5py.File(data_path)
    list(F.keys())
    CH1 = F[list(F.keys())[0]]['values']
    CH2 = F[list(F.keys())[1]]['values']
    CH3 = F[list(F.keys())[2]]['values']
    print('CH1', CH1.shape, 'CH2', CH2.shape, 'CH3', CH3.shape)

    # concatinating 3 channels
    signals = np.concatenate((CH1, CH2, CH3), axis=0)
    print(f'output signal size is: {signals.shape}')

    # making numpy array of signal and transposing it
    signals = np.array(signals.T)

    # re ordering channels
    signals = signals[:, orders]

    # adjusting signal length to label length
    nr_epochs = int(np.floor(signals.shape[0] / epoch_length))

    # warning if signal length is not equal to epoch_length * nr_epochs
    if signals.shape[0] != (nr_epochs * epoch_length):
        print('WARNING: signal length is not equal to number of epochs * epoch length')
        print('WARNING: signal will be cutted from end.')

    # cut-out signal ending which has no label
    signals = signals[0:int(nr_epochs * epoch_length), :]

    # selecting labels based on number of epochs
    labels = labels[0:nr_epochs]
    print(
        f'number of epochs is: {nr_epochs} and adjusted signals length is: {signals.shape[0]}')

    # getting number of classes
    nr_classes = len(np.unique(labels))
    print(f'number of unique classes are: {nr_classes}')

    # getting number of features (channels)
    nr_features = signals.shape[1]

    return signals, labels, epoch_length, nr_classes


def make_aux_data(data, epoch_length, labels):
    """
    This function reads 2d input data (time * features) and change it to
    3d strucure epochs * epoch_length * features

    data: A 2D matrix with shape [timesteps, feature]
    epoch_length: length of epoch
    labels: label value for all epochs
    """

    # import necessary packages
    import numpy as np

    # getting data dimension
    t, f = data.shape  # time * features

    # change epoch length to int
    epoch_length = int(epoch_length)

    # optimal number of epochs
    dim_0 = np.min(
        [np.int(np.floor(data.shape[0] / epoch_length)), len(labels)])
    print(f'optimal number of epochs are: {dim_0}')

    # initializing aux_data
    data_aux = np.zeros((dim_0, epoch_length, f), dtype=np.float16)
    print('auxilary data initial size', data_aux.shape)

    for i in range(dim_0):
        data_aux[i, :] = data[i * epoch_length:i * epoch_length + epoch_length]

    print('auxilary data final size', data_aux.shape,
          '  labels final size', labels.shape)
    if data_aux.shape[0] != len(labels):
        raise NameError('label length is different than batch nr in data')

    return data_aux, dim_0, t, f


def check_installed_packages():
    """
    using this function we are checking if user computer has all required packages
    """
    pass


# read large eeg data and chunk it to epochs in the dictionary format
def process_input_data(path_to_file, start_index, end_index, epoch_len, fr):

    # Getting the path and loading data using mne.io.read_raw (it automatically
    # detect file ext.)
    print("reading data")
    info = IO().read_raw(path_to_file)

    # loading data to memory
    data = info.get_data().T
    print("data is loaded!")
    # pdb.set_trace()
    # start chunking data
    data = data[start_index:end_index]

    # get data length after cutting
    data_len = len(data)

    # get possible number of epochs
    num_sample_per_epoch = epoch_len * fr
    num_of_epoch = data_len // num_sample_per_epoch

    if data_len % num_sample_per_epoch:
        print(
            f"Possible number of epochs is {num_of_epoch}. Last {data_len - num_of_epoch * num_sample_per_epoch} samples are not used.")

    # start creating list of dictionaries
    my_dict = []

    # initial info we are feeding in
    # index = i --> number of epochs
    # data = n * n_ch --> traces per channel for that epoch
    # histograms = hist * n_ch --> amplitude histogram per channel
    # spectrums = spectrum * n_ch --> power spectrum of signals
    print("Running step 1.")
    my_dict = [{"data": data[i*num_sample_per_epoch: (i + 1) * num_sample_per_epoch],
                "index":i} for i in range(num_of_epoch - 1)]

    print("Running step 2.")
    for dict_ in my_dict:

        # get epoch data
        temp_data = dict_["data"]

        # prepare hists and spectrums
        hists = []
        spectrums = []

        # start loop for channels
        for i in range(temp_data.shape[1]):

            FE = FeatureEng(data=temp_data[:, i], fs=fr)
            hists.append(FE.histogram())
            spectrums.append(FE.power_spec(keep_f=30))

        # load calculation to dictionary
        dict_.update({"histograms": hists,
                     "spectrums": spectrums,
                      "data": temp_data.T.tolist()})

    return my_dict


# write dictionary to json
def dict_to_json(path, input_dict):

    with open(path, "w") as f:
        json.dump(input_dict, f, cls=NumpyArrayEncoder)
        time.sleep(.01)


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
