import numpy as np
import pickle
from src.Utils import load_processed_data
from loguru import logger


def load_processed_data_inc_mean_std():
        dataset = load_processed_data()

        train_X = dataset['train_X'].astype('int32')
        val_X = dataset['val_X'].astype('int32')
        test_X = dataset['test_X'].astype('int32')

        test_y = dataset['test_y']
        val_y = dataset['val_y']
        train_y = dataset['train_y']
        label_to_names = dataset['label_to_names']

        def calc_mean_std(X):
            mean = np.mean(X)
            std = np.std(X)
            return mean, std

        # De data van train_X is genoeg om de mean en std van de hele set nauwkeurig te benaderen
        mean, std = calc_mean_std(train_X)

        model_ouput_extra = dict(
            mean=mean,
            std=std,
            train_X=train_X.astype('int32'),
            train_y=train_y,
            val_X=val_X.astype('int32'),
            val_y=val_y,
            test_X=test_X.astype('int32'),
            test_y=test_y,
            label_to_names=label_to_names, )

        output = open('data/processed/data_incl_mean_std.pkl', 'wb')
        pickle.dump(model_ouput_extra, output)
        output.close()
        logger.info('DONE with load_proces_incl_std_mean!')
        return mean, std, train_X, train_y, val_X, val_y, test_X, test_y, label_to_names

