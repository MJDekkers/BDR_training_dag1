from src.Utils import load_processed_data_inc_mean_std, conv_net, save_model
from loguru import logger

def train_model():
    dataset = load_processed_data_inc_mean_std()

    train_X = dataset['train_X'].astype('int32')
    val_X = dataset['val_X'].astype('int32')

    val_y = dataset['val_y']
    train_y = dataset['train_y']

    nr_classes = 10
    batch_size = 50
    epochs = 10


    model = conv_net(train_X, nr_classes)
    model.fit(x=train_X, y=train_y, batch_size=batch_size, epochs=epochs,
              validation_data=(val_X, val_y), verbose=2)
    #accuracy = get_results(model, 'report_train')
    #logger.info("Finished making predictions, accuracy = {}".format(accuracy))

    save_model(model)
    logger.info('DONE with train_data!')
    return model