from __future__ import print_function, division
import random
import sys

from matplotlib import rcParams
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import h5py
import tensorflow as tf

from keras.models import load_model
# from keras.models import Sequential
# from keras.layers import Dense, Conv1D, LSTM, Bidirectional, Dropout
# from keras.utils import plot_model

from nilmtk.utils import find_nearest
from nilmtk.feature_detectors import cluster
from nilmtk.legacy.disaggregate import Disaggregator
from nilmtk.datastore import HDFDataStore

class RNNDisaggregator(Disaggregator):

    def __init__(self):
        '''Initialize disaggregator
        '''
        self.MODEL_NAME = "RNN"
        self.mmax = None
        self.MIN_CHUNK_LENGTH = 60
        self.model = self._create_model()

    def replace_below_50(self, generator):
        for value in generator:
            if value < 30:
                yield 0
            else:
                yield value
    
    def array_to_generator(self, data, chunk_size):
        for i in range(0, len(data), chunk_size):
            yield data[i:i+chunk_size]



    def train(self, mains, meter, epochs=1, batch_size=60, **load_kwargs):

        # get_max = maxi
        history = {'loss': [], 'val_loss': []}

        # A = mains.power_series()
        # B = meter.power_series()
        # C = next(A)
        # D = next(B)
        # ix = C.index.intersection(D.index)
        # E = np.array(C[ix])
        # F = np.array(D[ix])

        # G = self.array_to_generator(mains,50000)
        # print("abcdefg")
        # H = self.array_to_generator(meter,50000)

        


        # main_power_series = G
        # meter_power_series = H
        main_power_series = mains.power_series(**load_kwargs)
        meter_power_series = meter.power_series(**load_kwargs)

        # self.replace_below_50(meter_power_series)
        # Train chunks
        run = True
        mainchunk = next(main_power_series)
        meterchunk = next(meter_power_series)
        if self.mmax == None:
            print('get max')
            self.mmax = mainchunk.max()

        while(run):
            mainchunk = self._normalize(mainchunk, self.mmax)
            meterchunk = self._normalize(meterchunk, self.mmax)

            history_chunk = self.train_on_chunk(mainchunk, meterchunk, epochs, batch_size)
            history['loss'].extend(history_chunk.history['loss'])
            history['val_loss'].extend(history_chunk.history['val_loss'])
            try:
                mainchunk = next(main_power_series)
                meterchunk = next(meter_power_series)
            except:
                run = False
        return history

    def train_on_chunk(self, mainchunk, meterchunk, epochs, batch_size):

        # Replace NaNs with 0s
        mainchunk = np.nan_to_num(mainchunk, nan=0)
        meterchunk = np.nan_to_num(meterchunk, nan=0)
        # ix = mainchunk.index.intersection(meterchunk.index)
        # mainchunk = np.array(mainchunk[ix])
        # meterchunk = np.array(meterchunk[ix])

        mainchunk = np.reshape(mainchunk, (mainchunk.shape[0],1,1))

        history = self.model.fit(mainchunk, meterchunk, epochs=epochs, batch_size=batch_size, shuffle=True,validation_split = 0.2)
        return history

    def array_to_generator_dis(self,data):
        for i in data:
                yield i
    def disaggregate(self, mains, output_datastore, meter_metadata, **load_kwargs):

        load_kwargs = self._pre_disaggregation_checks(load_kwargs)

        load_kwargs.setdefault('sample_period', 60)
        # load_kwargs.setdefault('sections', mains.good_sections())
        load_kwargs.setdefault('sections', None)

        timeframes = []
        building_path = '/building{}'.format(mains.building())
        mains_data_location = building_path + '/elec/meter1'
        data_is_available = False

        for chunk in mains.power_series(**load_kwargs):
            if len(chunk) < self.MIN_CHUNK_LENGTH:
                continue
            print("New sensible chunk: {}".format(len(chunk)))

            timeframes.append(chunk.timeframe)
            measurement = chunk.name
            chunk2 = self._normalize(chunk, self.mmax)

            appliance_power = self.disaggregate_chunk(chunk2)
            appliance_power[appliance_power < 0] = 0
            appliance_power = self._denormalize(appliance_power, self.mmax)

            # Append prediction to output
            data_is_available = True
            cols = pd.MultiIndex.from_tuples([chunk.name])
            meter_instance = meter_metadata.instance()
            df = pd.DataFrame(
                appliance_power.values, index=appliance_power.index,
                columns=cols, dtype="float32")
            key = '{}/elec/meter{}'.format(building_path, meter_instance)
            output_datastore.append(key, df)

            # Append aggregate data to output
            mains_df = pd.DataFrame(chunk, columns=cols, dtype="float32")
            output_datastore.append(key=mains_data_location, value=mains_df)
        

        # Save metadata to output
        if data_is_available:
            self._save_metadata_for_disaggregation(
                output_datastore=output_datastore,
                sample_period=load_kwargs['sample_period'],
                measurement=measurement,
                timeframes=timeframes,
                building=mains.building(),
                meters=[meter_metadata]
            )

    def disaggregate_chunk(self, mains):
        up_limit = len(mains)

        mains.fillna(0, inplace=True)

        X_batch = np.array(mains)
        X_batch = np.reshape(X_batch, (X_batch.shape[0],1,1))

        pred = self.model.predict(X_batch, batch_size=128)
        pred = np.reshape(pred, (len(pred)))
        column = pd.Series(pred, index=mains.index[:len(X_batch)], name=0)

        appliance_powers_dict = {}
        appliance_powers_dict[0] = column
        appliance_powers = pd.DataFrame(appliance_powers_dict)
        return appliance_powers

    def import_model(self, filename):

        self.model = load_model(filename)
        with h5py.File(filename, 'a') as hf:
            ds = hf.get('disaggregator-data').get('mmax')
            self.mmax = np.array(ds)[0]

    def export_model(self, filename):

        self.model.save(filename)
        with h5py.File(filename, 'a') as hf:
            gr = hf.create_group('disaggregator-data')
            gr.create_dataset('mmax', data = [self.mmax])

    def _normalize(self, chunk, mmax):

        tchunk = chunk / mmax
        return tchunk

    def _denormalize(self, chunk, mmax):

        tchunk = chunk * mmax
        return tchunk

    def _create_model(self):

        model = tf.keras.models.Sequential()

        # 1D Conv
        model.add(tf.keras.layers.Conv1D(16, 4, activation="linear", input_shape=(1,1), padding="same", strides=1))

        #Bi-directional LSTMs
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, stateful=False), merge_mode='concat'))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=False, stateful=False), merge_mode='concat'))

        # Fully Connected Layers
        model.add(tf.keras.layers.Dense(128, activation='tanh'))
        model.add(tf.keras.layers.Dense(1, activation='linear'))

        model.compile(loss='mse', optimizer='adam')
        model.summary()

        return model