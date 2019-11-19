import os
import tarfile
import pickle
import numpy as np
from urllib.request import urlretrieve


def load_data():
    # training set, batches 1-4
    if not os.path.exists(os.path.join(os.getcwd(), "data/raw")):
        os.makedirs(os.path.join(os.getcwd(), "data/raw"))

    dataset_dir = os.path.join(os.getcwd(), "data/raw")

    if not os.path.exists(os.path.join(dataset_dir, "cifar-10-batches-py")):
        print("Downloading data...")
        urlretrieve("http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
                    os.path.join(dataset_dir, "cifar-10-python.tar.gz"))
        tar = tarfile.open(os.path.join(dataset_dir, "cifar-10-python.tar.gz"))
        tar.extractall(dataset_dir)
        tar.close()

    train_X = np.zeros((40000, 3, 32, 32), dtype="float32")
    train_y = np.zeros((40000, 1), dtype="ubyte").flatten()
    n_samples = 10000  # aantal samples per batch
    dataset_dir = os.path.join(dataset_dir, "cifar-10-batches-py")
    for i in range(0, 4):
        f = open(os.path.join(dataset_dir, "data_batch_" + str(i + 1)), "rb")
        cifar_batch = pickle.load(f, encoding="latin1")
        f.close()
        train_X[i * n_samples:(i + 1) * n_samples] = (cifar_batch['data'].reshape(-1, 3, 32, 32) / 255.).astype(
            "float32")
        train_y[i * n_samples:(i + 1) * n_samples] = np.array(cifar_batch['labels'], dtype='ubyte')

    # validation set, batch 5
    f = open(os.path.join(dataset_dir, "data_batch_5"), "rb")
    cifar_batch_5 = pickle.load(f, encoding="latin1")
    f.close()
    val_X = (cifar_batch_5['data'].reshape(-1, 3, 32, 32) / 255.).astype("float32")
    val_y = np.array(cifar_batch_5['labels'], dtype='ubyte')

    # labels
    f = open(os.path.join(dataset_dir, "batches.meta"), "rb")
    cifar_dict = pickle.load(f, encoding="latin1")
    label_to_names = {k: v for k, v in zip(range(10), cifar_dict['label_names'])}
    f.close()

    # test set
    f = open(os.path.join(dataset_dir, "test_batch"), "rb")
    cifar_test = pickle.load(f, encoding="latin1")
    f.close()
    test_X = (cifar_test['data'].reshape(-1, 3, 32, 32) / 255.).astype("float32")
    test_y = np.array(cifar_test['labels'], dtype='ubyte')

    cifar_output = dict(
        train_X=train_X.astype('int32'),
        train_y=train_y,
        val_X=val_X.astype('int32'),
        val_y=val_y,
        test_X =test_X.astype('int32'),
        test_y=test_y,
        label_to_names=label_to_names,)

    output = open('data/processed/cifar-10.pkl', 'wb')
    pickle.dump(cifar_output, output)
    output.close()




