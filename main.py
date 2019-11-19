from loguru import logger
from src.data.Load_data import load_data
from src.features.build_features import load_processed_data_inc_mean_std, load_mean_std
from src.models.Train_model import train_model
from src.models.Predict_model import predict_model

def main():
    load_data()
    load_mean_std()
    #load_processed_data_inc_mean_std()
    train_model()
    predict_model()
    logger.info('DONE with main_prepare!')
    return

if __name__ == '__main__':
     main()
