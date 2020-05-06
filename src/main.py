"""
Created on Mon Mar 10 15:59:47 2020

@author: harsh
"""
import argparse
import torch
import gc
import os
import configparser

from average_loss.AverageLoss import AverageLoss
from data.CustomDataset import CustomDataset
from data.Dataloader import Dataloader
from general_utils import generate_sample
from models.Hybrid_L1_Model import Hybrid_L1_Model
from models.Hybrid_L2_Model import Hybrid_L2_Model
from models.L1_Model import L1_Model
from models.L2_Model import L2_Model


def get_dataloader(dataset_path, image_format, image_size, batch_size, config=None):
    print('Setting up Dataloader for the given dataset!')

    dataset_list_train = [config.get('DataloaderSection', 'train_dataset_X'),
                          config.get('DataloaderSection', 'train_dataset_Y')]
    dataset_list_test = [config.get('DataloaderSection', 'test_dataset_X'),
                         config.get('DataloaderSection', 'test_dataset_Y')]

    trainloaders = Dataloader(dataset_path, image_size, train=True, image_format=image_format,
                              batch_size=batch_size).get_data_loader(dataset_list_train)
    testloaders = Dataloader(dataset_path, image_size, train=False, image_format=image_format,
                             batch_size=1).get_data_loader(
        dataset_list_test)

    return trainloaders, testloaders


def get_model_params(config):
    # Get additional parameters to initialize model
    epochs = int(config.get('ModelSection', 'epochs'))
    if epochs < 0:
        raise Exception('Invalid value of number of epochs!')
    lr = float(config.get('ModelSection', 'learning-rate'))
    if lr < 0:
        raise Exception('Invalid learning rate!')
    leaky_thresh = float(config.get('ModelSection', 'leaky_thresh'))
    lamda = int(config.get('ModelSection', 'lamda'))
    if lamda < 0:
        raise Exception('Incorrect value of Lambda!')
    beta1 = float(config.get('ModelSection', 'beta1'))
    beta2 = float(config.get('ModelSection', 'beta2'))
    if beta1 < 0 or beta2 < 0 or beta1 > 1 or beta2 > 1:
        raise Exception('Incorrect Values of beta!')

    return epochs, lr, leaky_thresh, lamda, beta1, beta2


def initialize_model(global_path, image_size, image_format, config, loss_type):
    model = None
    torch.cuda.empty_cache()
    gc.collect()

    epochs, lr, leaky_thresh, lamda, beta1, beta2 = get_model_params(config)

    if loss_type == 'hybrid_l1':
        model = Hybrid_L1_Model(base_path=global_path, image_size=image_size, image_format=image_format, epochs=epochs,
                                learning_rate=lr, leaky_relu=leaky_thresh, lamda=lamda, betas=(beta1, beta2))
    elif loss_type == 'hybrid_l2':
        model = Hybrid_L2_Model(base_path=global_path, image_size=image_size, image_format=image_format, epochs=epochs,
                                learning_rate=lr, leaky_relu=leaky_thresh, lamda=lamda, betas=(beta1, beta2))
    elif loss_type == 'l1':
        model = L1_Model(base_path=global_path, image_size=image_size, image_format=image_format, epochs=epochs,
                         learning_rate=lr, leaky_relu=leaky_thresh, lamda=lamda, betas=(beta1, beta2))
    elif loss_type == 'l2':
        model = L2_Model(base_path=global_path, image_size=image_size, image_format=image_format, epochs=epochs,
                         learning_rate=lr, leaky_relu=leaky_thresh, lamda=lamda, betas=(beta1, beta2))
    else:
        raise NotImplementedError('This Loss function has not been implemented!')

    average_loss = AverageLoss(os.path.join(global_path, 'Loss_Checkpoints'),
                               models=(4, ['Generator_X', 'Generator_Y', 'Discriminator_X', 'Discriminator_Y']))

    return model, average_loss


def load_model():
    epochs, lr, leaky_thresh, lamda, beta1, beta2 = get_model_params(config)
    ckp_name = config.get('LoadModelSection', 'checkpoint_name')
    avg_ckp_name = config.get('LoadModelSection', 'average_ckp_name')
    avg_ckp_index = int(config.get('LoadModelSection', 'average_ckp_index'))
    model.load_checkpoint(ckp_name)
    average_loss.load(avg_ckp_name, avg_ckp_index)
    model.set_all_params(epochs, lr, leaky_thresh, lamda, (beta1, beta2))


def train_model(model, trainloaders, testloaders, average_loss, epochs):
    evaluate = config.get('ModelTrainingSection', 'evaluate') == 'True'
    eval = (False, None, None)
    if evaluate and testloaders is not None:
        eval_epochs = int(config.get('ModelTrainingSection', 'evaluate_after_epochs'))
        if eval_epochs < 0:
            raise Exception('Incorrect value of evaluation epochs!')
        eval = (True, testloaders, eval_epochs)

    save_model = config.get('ModelTrainingSection', 'save_model') == 'True'
    save = (False, 1)
    if save_model:
        save_after_epoch = config.get('ModelTrainingSection', 'save_after_epochs')
        save_epochs = int(save_after_epoch)
        if save_epochs < 0:
            raise Exception('Incorrect value of Save epochs!')
        save = (True, save_epochs)

    display_test_img = (False, None, 25)
    display_test_img_flag = config.get('ModelTrainingSection', 'display_test_image') == 'True'
    if display_test_img_flag and testloaders is not None:
        display_epochs = config.get('ModelTrainingSection', 'display_test_image_epochs')
        display_img_epochs = int(display_epochs)
        if display_img_epochs < 0:
            raise Exception('Incorrect value of Display Image epochs!')
        display_test_img = (True, testloaders, display_img_epochs)

    if epochs is not None:
        epochs = int(epochs)
        model.change_params(epochs=epochs)

    smoothen_flag = config.get('ModelTrainingSection', 'smoothen') == 'True'

    model.train_model(trainloaders, average_loss, eval=eval, display_test_image=display_test_img,
                      save_model=save, smoothen=smoothen_flag)
    average_loss.plot()


def evaluate_model(model, loader_X, loader_Y, config):
    samples = int(config.get('ModelEvaluationSection', 'no_samples'))
    if samples < 0:
        raise Exception('Invalid value of number of samples!')
    model.evaluate_model(loader_X, 'X', evaluate_X=True, no_of_images=samples)
    model.evaluate_model(loader_Y, 'Y', evaluate_X=False, no_of_images=samples)
    model.evaluate_L1_loss_dataset(loader_X, evaluate_X=True)
    model.evaluate_L1_loss_dataset(loader_Y, evaluate_X=False)


def apply_model(base_path, folder, model, eval_X=True, image_size=256, image_format='jpg'):
    dataset = CustomDataset('{}/{}'.format(base_path, folder), image_size, image_format)
    loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=1)
    converted_path = '{}/{}_converted'.format(base_path, folder)
    try:
        if not os.path.exists(converted_path):
            os.makedirs(converted_path)
    except OSError:
        print('Error: Creating directory of data')
    model.run_model_on_dataset(loader, '{}_converted'.format(folder), save_path=base_path, evaluate_X=eval_X)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Defining the GAN parameters and training the model!')
    parser.add_argument('-dpath', metavar='dataset-path', action='store', required=True,
                        help='The base path of the dataset folder')
    parser.add_argument('-bpath', metavar='base-path', action='store', required=True,
                        help='The base path where checkpoints and evaluation data would be saved')
    parser.add_argument('-folder', metavar='folder', action='append', required=True,
                        help='The folder to be used inside the base path')
    parser.add_argument('-format', metavar='Image Format', action='store', default='jpg',
                        help='Image format to be considered')
    parser.add_argument('-size', metavar='Image Size', action='store', default=256, help='Image size to be considered')
    parser.add_argument('-batch', metavar='Batch Size', action='store', required=True,
                        help='Batch size to be used in training set')
    parser.add_argument('-epochs', metavar='Epochs', action='store', default=None, help='No of Epochs for training. '
                                                                                        'Overrides the parameter '
                                                                                        'present in properties file')
    parser.add_argument('-load_model', action='store_true', default=False, help='Use this option to load a model')
    parser.add_argument('-run_model', action='store_true', default=False, help='Use this option to run a model on a '
                                                                               'dataset. Loads the model saved at '
                                                                               'checkpoint.')
    parser.add_argument('-evaluate', action='store_true', default=False, help='Evaluate the model only')
    parser.add_argument('-loss_type', metavar='Loss Type', action='store', default='l1',
                        help='The Loss function to be used. '
                             'Options are - hybrid_l1, hybrid_l2, '
                             'l1, l2')
    parser.add_argument('-mtype', metavar='Model Type', action='store', default='unet',
                        help='The model architecture to be initialized')

    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read('./project.properties')

    base_path = args.bpath
    for paths in args.folder:
        base_path = os.path.join(base_path, paths)

    layer_size = int(config.get('ModelSection', 'layer_size'))
    option = {}
    option['lr_policy'] = config.get('ModelTrainingSection', 'lr_policy')
    option['n_epochs'] = int(config.get('ModelTrainingSection', 'n_epochs'))
    option['n_epoch_decay'] = int(config.get('ModelTrainingSection', 'n_epoch_decay'))
    option['step_size'] = int(config.get('ModelTrainingSection', 'step_size'))

    model, average_loss = initialize_model(base_path, args.size, args.format, config, args.loss_type)
    if args.load_model or args.run_model or args.evaluate:
        load_model()
    else:
        residual_blocks = int(config.get('ModelSection', 'resnet.residual_blocks'))
        model.initialize_model(lr_schedular_options=option, model_type=args.mtype, residual_blocks=residual_blocks,
                               layer_size=layer_size)

    if args.run_model:
        folder = config.get('LoadModelSection', 'save_folder')
        evaluate_X = config.get('LoadModelSection', 'evaluate_X_run_model_param') == 'True'
        apply_model(args.dpath, folder, model, evaluate_X, args.size, args.format)
        print('Task Completed!')
        exit()

    trainloaders, testloaders = get_dataloader(args.dpath, args.format, args.size, int(args.batch), config)
    generate_sample(trainloaders)

    if args.evaluate:
        evaluate_model(model, testloaders[0], testloaders[1], config)
        exit()

    train_model(model, trainloaders, testloaders, average_loss, args.epochs)

    evaluate_model_performance = config.get('ModelEvaluationSection', 'evaluate_model') == 'True'
    if evaluate_model_performance:
        evaluate_model(model, testloaders[0], testloaders[1], config)
