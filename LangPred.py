"""Machine learning model for programming identification"""

from distutils.sysconfig import EXEC_PREFIX
import os
import gc
import logging
import random
from pathlib import Path
from math import ceil
import json
import warnings

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE , ADASYN
from collections import Counter


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

from FeatureExtract import extract, CONTENT_SIZE

from Proccess import (search_files, extract_from_files, read_file)

# Settings list
# LOGGER = logging.getLogger(__name__)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings("ignore")


# /////////////////////////original 
'''
_NEURAL_NETWORK_HIDDEN_LAYERS = [256, 64, 16]
_OPTIMIZER_STEP = 0.05

_FITTING_FACTOR = 20
_CHUNK_PROPORTION = 0.2
_CHUNK_SIZE = 1000
'''
# //////////////////////////

_NEURAL_NETWORK_HIDDEN_LAYERS = [2]
_OPTIMIZER_STEP = 0.05

_FITTING_FACTOR = 20
_CHUNK_PROPORTION = 0.2
_CHUNK_SIZE = 2000


class Predictor:

    def __init__(self, model_dir=os.curdir, lang_json='languages.json'):

        # trained model dir
        self.model_dir = model_dir

        #: tells if current model is the default model
        # self.is_default = model_data[1]

        #: supported languages with associated extensions
        with open(lang_json) as f:
            self.languages = json.load(f)

        n_classes = len(self.languages)

        feature_columns = [tf.feature_column.numeric_column('', (CONTENT_SIZE,))]

        self._classifier = tf.compat.v1.estimator.DNNLinearCombinedClassifier(
            linear_feature_columns=feature_columns,
            dnn_feature_columns=feature_columns,
            dnn_hidden_units=_NEURAL_NETWORK_HIDDEN_LAYERS,
            n_classes=n_classes,
            linear_optimizer=tf.compat.v1.train.RMSPropOptimizer(_OPTIMIZER_STEP),
            dnn_optimizer=tf.compat.v1.train.RMSPropOptimizer(_OPTIMIZER_STEP),
            model_dir=self.model_dir,
        )

    def language(self, text):
        # predict language name
        values = extract(text)
        input_fn = _to_func([[values], []])
        proba = next(self._classifier.predict(input_fn=input_fn))
        # proba = proba.tolist()
        proba = proba['probabilities']
        # print(proba)
        # Order the languages from the most probable to the least probable
        positions = np.argsort(proba)[::-1]
        # print(positions)
        names = np.sort(list(self.languages))
        # print(names)
        # print(self.languages)
        names = names[positions]
        
        return names[0]

    def learn(self, input_dir):
        """Learning model"""

        extensions = [ext for exts in self.languages.values() for ext in exts]
        print(extensions)
        files = search_files(input_dir, extensions)
        random.shuffle(files)
        files = files[:]

        test_idx = int(len(files) * 0.2)
        evaluation_paths = files[:test_idx]
        train_paths_ = files[test_idx:]
        train_paths = []

        nb_files = len(train_paths_)
        chunk_size = min(int(_CHUNK_PROPORTION * nb_files), _CHUNK_SIZE)
        batches = _pop_many(train_paths_, chunk_size)
        print('chunk size : ',chunk_size)
        print(type(train_paths_))
        evaluation_data = extract_from_files(evaluation_paths, self.languages)
        print("Evaluation data class count :",Counter(evaluation_data[1]))
        print('evaluation_data : ',type(evaluation_data[0]) )
        print('evaluation_data : ',len(evaluation_data))
        print('evaluation_data : ',evaluation_data[1].shape )
        print("Start learning")
        # return

        # ///////////////////////////////////////////

        for training_files in batches:
           
            train_paths.extend(training_files)
            training_data = extract_from_files(training_files, self.languages)
            training_data = list(training_data)
            smote = SMOTE(random_state=130)
            print(len(training_data[0]),len(training_data[1]),Counter(training_data[1]))
            try:
                training_data = smote.fit_resample(training_data[0],training_data[1])
            except Exception as e:
                pass
                print("error in oversampling")
            # print(training_data)

            steps = int(_FITTING_FACTOR * len(training_data[0]) / 100)
            if steps == 0:
                break

            self._classifier.train(input_fn=_to_func(training_data), steps=steps)

            # evaluation

        accuracy = self._classifier.evaluate(
            input_fn=_to_func(evaluation_data), steps=1)['accuracy']
        print(accuracy)


        return train_paths, evaluation_paths


def _pop_many(items, chunk_size):
    while items:
        yield items[0:chunk_size]

        # Avoid memory overflow
        del items[0:chunk_size]
        gc.collect()


def _to_func(vector):
    return lambda: (
        {'': tf.constant(vector[0], name='const_features')},
        tf.constant(vector[1], name='const_labels'))
