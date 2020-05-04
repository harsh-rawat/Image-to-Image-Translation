from abc import ABC, abstractmethod
import pathlib
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import torchvision.models as models
import torch.optim.lr_scheduler as lr_schedular

from models.Discriminator import Discriminator
from models.Generator_Fusion import Generator_Fusion
from models.Generator_RESNET import Generator_RESNET
from models.Generator_Unet import Generator_Unet
from models.Generator_Unet_Fusion import Generator_Unet_Fusion
from models.Generator_Unet_Shallow import Generator_Unet_Shallow


class Model(ABC):

    def __init__(self, base_path='', epochs=10, learning_rate=0.0002, image_size=256, leaky_relu=0.2,
                 betas=(0.5, 0.999), lamda=100, image_format='png'):
        self.image_size = image_size
        self.leaky_relu_threshold = leaky_relu

        self.epochs = epochs
        self.lr = learning_rate
        self.betas = betas
        self.lamda = lamda
        self.base_path = base_path
        self.image_format = image_format
        self.count = 1

        self.genX = None
        self.genY = None
        self.disX = None
        self.disY = None
        self.gen_optim = None
        self.disX_optim = None
        self.disY_optim = None
        self.model_type = None

        vgg16 = models.vgg16()
        self.vgg16_conv = nn.Sequential(*list(vgg16.children())[:-3])

        self.residual_blocks = 9
        self.layer_size = 64
        self.lr_policy = None
        self.lr_schedule_gen = None
        self.lr_schedule_disX = None
        self.lr_schedule_disY = None

        self.device = self.get_device()
        self.create_folder_structure()

    def create_folder_structure(self):
        checkpoint_folder = self.base_path + '/checkpoints'
        loss_folder = self.base_path + '/Loss_Checkpoints'
        training_folder = self.base_path + '/Training Images'
        test_folder = self.base_path + '/Test Images'
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        if not os.path.exists(loss_folder):
            os.makedirs(loss_folder)
        if not os.path.exists(training_folder):
            os.makedirs(training_folder)
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)

    def get_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        print(torch.cuda.get_device_name(0))

        if device.type == 'cuda':
            print('Memory Usage -')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
            return device
        else:
            return None

    def initialize_model(self, lr_schedular_options, model_type='unet', residual_blocks=9, layer_size=64):

        all_models = ['unet', 'resnet', 'fusion', 'unet_fusion', 'unet_shallow']
        if model_type not in all_models:
            raise Exception('This model type is not available!');

        self.disX = Discriminator(image_size=self.image_size, leaky_relu=self.leaky_relu_threshold)
        self.disY = Discriminator(image_size=self.image_size, leaky_relu=self.leaky_relu_threshold)

        if model_type == 'unet':
            self.genX = Generator_Unet(image_size=self.image_size, ngf=layer_size)
            self.genY = Generator_Unet(image_size=self.image_size, ngf=layer_size)
        elif model_type == 'resnet':
            self.genX = Generator_RESNET(residual_blocks=residual_blocks, ngf=layer_size)
            self.genY = Generator_RESNET(residual_blocks=residual_blocks, ngf=layer_size)
        elif model_type == 'fusion':
            self.genX = Generator_Fusion(ngf=layer_size)
            self.genY = Generator_Fusion(ngf=layer_size)
        elif model_type == 'unet_fusion':
            self.genX = Generator_Unet_Fusion(image_size=self.image_size, ngf=layer_size)
            self.genY = Generator_Unet_Fusion(image_size=self.image_size, ngf=layer_size)
        elif model_type == 'unet_shallow':
            self.genX = Generator_Unet_Shallow(image_size=self.image_size, ngf=layer_size)
            self.genY = Generator_Unet_Shallow(image_size=self.image_size, ngf=layer_size)

        if self.device is not None:
            self.genX.cuda()
            self.disX.cuda()
            self.genY.cuda()
            self.disY.cuda()

        gen_params = list(self.genX.parameters()) + list(self.genY.parameters())

        self.gen_optim = optim.Adam(gen_params, lr=self.lr, betas=self.betas)
        self.disX_optim = optim.Adam(self.disX.parameters(), lr=self.lr, betas=self.betas)
        self.disY_optim = optim.Adam(self.disY.parameters(), lr=self.lr, betas=self.betas)

        self.lr_schedule_disX = self.get_learning_schedule(self.disX_optim, lr_schedular_options)
        self.lr_schedule_disY = self.get_learning_schedule(self.disY_optim, lr_schedular_options)
        self.lr_schedule_gen = self.get_learning_schedule(self.gen_optim, lr_schedular_options)

        self.model_type = model_type
        self.layer_size = layer_size
        self.residual_blocks = residual_blocks
        self.lr_policy = lr_schedular_options
        print('Model Initialized !\nGenerator Model Type : {} and Layer Size : {}'.format(model_type, layer_size))
        print('Model Parameters are:\nEpochs : {}\nLearning rate : {}\nLeaky Relu Threshold : {}\nLamda : {}\nBeta : {}'
              .format(self.epochs, self.lr, self.leaky_relu_threshold, self.lamda, self.betas))

    @abstractmethod
    def calculate_image_similarity_loss(self, img1, img2):
        pass

    def train_model(self, trainloader, average_loss, eval=(False, None, None), save_model=(False, 25),
                    display_test_image=(False, None, 25)):

        train_X = trainloader[0]
        train_Y = trainloader[1]

        mean_loss = nn.BCELoss()

        batches_X_train = len(train_X)
        batches_Y_train = len(train_Y)

        self.disX.train()
        self.disY.train()
        self.genX.train()
        self.genY.train()

        sample_img_test = None
        if display_test_image[0]:
            sample_img_test = next(iter(display_test_image[1][0]))  # X Sample Image
            save_image((sample_img_test[0].detach().cpu() + 1) / 2,
                       '{}/Training Images/orig_img_X.{}'.format(self.base_path, self.image_format))
            if self.device is not None:
                sample_img_test = sample_img_test.cuda()

        batches = min(batches_X_train, batches_Y_train)
        iter_count = 0

        print('Batches in a Epoch are : {}'.format(batches))

        for i in range(self.epochs):
            if eval[0] and (i % eval[2] == 0):
                self.evaluate_L1_loss_dataset(eval[1][0], evaluate_X=True)
                self.evaluate_L1_loss_dataset(eval[1][1], evaluate_X=False)
                self.genX.train()
                self.genY.train()

            gen_X_loss = 0
            gen_Y_loss = 0
            dis_X_loss = 0
            dis_Y_loss = 0

            count_train = 0

            iter_data_X = iter(train_X)
            iter_data_Y = iter(train_Y)

            while count_train < batches:

                imgs_X = next(iter_data_X)
                imgs_Y = next(iter_data_Y)
                count_train += 1

                if len(imgs_X) != len(imgs_Y):
                    continue

                batch_size = len(imgs_X)
                zero_label = torch.zeros(batch_size)
                one_label = torch.ones(batch_size)

                if self.device is not None:
                    imgs_X = imgs_X.cuda()
                    imgs_Y = imgs_Y.cuda()
                    zero_label = zero_label.cuda()
                    one_label = one_label.cuda()

                # Update Discriminator X
                self.dis_X_optim.zero_grad()

                fake_images_X = self.genY(imgs_Y)

                loss_fake_disX = mean_loss(self.disX(fake_images_X), zero_label)
                loss_real_disX = mean_loss(self.disX(imgs_X), one_label)

                loss_disX = loss_fake_disX + loss_real_disX
                loss_disX.backward()
                self.dis_X_optim.step()

                # Discriminator Y update - Real and Generated Images
                self.dis_Y_optim.zero_grad()

                fake_images_Y = self.genX(imgs_X)
                loss_fake_disY = mean_loss(self.disY(fake_images_Y), zero_label)
                loss_real_disY = mean_loss(self.disY(imgs_Y), one_label)

                loss_disY = loss_fake_disY + loss_real_disY
                loss_disY.backward()
                self.dis_Y_optim.step()

                # Generator Updates : Generator loss on real images and cyclic consistency check

                self.gen_optim.zero_grad()

                # Update for Generator X
                fake_images_Y_2 = self.genX(imgs_X)
                loss_genX_fake = mean_loss(self.disY(fake_images_Y_2), one_label)
                loss_cyclic_genX = self.calculate_image_similarity_loss(self.genY(fake_images_Y_2), imgs_X)

                # Update for Generator Y
                fake_images_X_2 = self.genY(imgs_Y)
                loss_genY_fake = mean_loss(self.disX(fake_images_X_2), one_label)
                loss_cyclic_genY = self.calculate_image_similarity_loss(self.genX(fake_images_X_2), imgs_Y)

                loss_total_gen = loss_genX_fake + loss_genY_fake + self.lamda * (loss_cyclic_genX + loss_cyclic_genY)
                loss_total_gen.backward()
                self.gen_optim.step()

                gen_X_loss += loss_genX_fake.item() + self.lamda * loss_cyclic_genX.item()
                gen_Y_loss += loss_genY_fake.item() + self.lamda * loss_cyclic_genY.item()
                dis_X_loss += loss_disX.item()
                dis_Y_loss += loss_disY.item()

                iter_count += 1

            gen_Y_loss /= (iter_count * 1.0)
            gen_X_loss /= (iter_count * 1.0)
            dis_X_loss /= (iter_count * 1.0)
            dis_Y_loss /= (iter_count * 1.0)

            print('Epoch : {}, Generator X Loss : {}, Generator Y Loss : {}, Discriminator X Loss : {}, Discriminator '
                  'Y Loss : {}'.format(i + 1, gen_X_loss, gen_Y_loss, dis_X_loss, dis_Y_loss))

            if display_test_image[0] and i % display_test_image[2] == 0:
                self.genX.eval()
                out_result = self.genX(sample_img_test)
                out_result = out_result.detach().cpu()
                out_result = (out_result[0] + 1) / 2
                save_image(out_result, '{}/Training Images/epoch_{}.{}'.format(self.base_path, i,
                                                                               self.image_format))
                self.genX.train()

            save_tuple = ([gen_X_loss], [gen_Y_loss], [dis_X_loss], [dis_Y_loss])
            average_loss.add_loss(save_tuple)

            if save_model[0] and i % save_model[1] == 0:
                self.save_checkpoint('checkpoint_epoch_{}'.format(i), self.model_type)
                average_loss.save('checkpoint_avg_loss', save_index=0)

            self.lr_schedule_gen.step()
            self.lr_schedule_disX.step()
            self.lr_schedule_disY.step()
            for param_grp in self.disX_optim.param_groups:
                print('Learning rate after {} epochs is : {}'.format(i + 1, param_grp['lr']))

        self.save_checkpoint('checkpoint_train_final', self.model_type)
        average_loss.save('checkpoint_avg_loss_final', save_index=0)

    def get_learning_schedule(self, optimizer, option):

        schedular = None
        if option['lr_policy'] == 'linear':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch - option['n_epochs']) / float(option['n_epoch_decay'] + 1)
                return lr_l

            schedular = lr_schedular.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif option['lr_policy'] == 'plateau':
            schedular = lr_schedular.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        elif option['lr_policy'] == 'step':
            schedular = lr_schedular.StepLR(optimizer, step_size=option['step_size'], gamma=0.1)
        elif option['lr_policy'] == 'cosine':
            schedular = lr_schedular.CosineAnnealingLR(optimizer, T_max=option['n_epochs'], eta_min=0)
        else:
            raise Exception('LR Policy not implemented!')

        return schedular

    def run_model_on_dataset(self, loader, save_folder, evaluate_X=True, save_path=None):
        if self.genX is None or self.genY is None or self.disX is None or self.disY is None:
            raise Exception('Model has not been initialized and hence cannot be saved!')
        index = 1
        if save_path is None:
            save_path = self.base_path
        for img in loader:

            if self.device is not None:
                img = img.cuda()

            filename = '{}/{}/{}.{}'.format(save_path, save_folder, index, self.image_format)
            index += 1

            output = None
            if evaluate_X:
                self.genX.eval()
                output = self.genX(img)
            else:
                self.genY.eval()
                output = self.genY(img)

            output = output[0].detach().cpu()
            output = (output + 1) / 2

            save_image(output, filename)

    def evaluate_model(self, loader, save_filename, evaluate_X=True, no_of_images=1):
        # Considering that we have batch size of 1 for test set
        if self.genX is None or self.genY is None or self.disX is None or self.disY is None:
            raise Exception('Model cannot be saved as it has not been initialized!')

        counter_images_generated = 0
        while counter_images_generated < no_of_images:
            img_test = next(iter(loader))

            if self.device is not None:
                img_test = img_test.cuda()

            filename = '{}/Test Images/{}_{}.{}'.format(self.base_path, save_filename, self.count, self.image_format)
            real_filename = '{}/Test Images/{}_{}_real.{}'.format(self.base_path, save_filename, self.count,
                                                                  self.image_format)

            output = None
            if evaluate_X:
                self.genX.eval()
                output = self.genX(img_test)
            else:
                self.genY.eval()
                output = self.genY(img_test)

            output = output[0].detach().cpu()
            output = (output + 1) / 2

            save_image(output, filename)
            save_image((img_test[0] + 1) / 2, real_filename)

            counter_images_generated += 1

    def get_closeness_score(self, real_img, test_img):
        return torch.mean(torch.abs(real_img - test_img))

    def evaluate_L1_loss_dataset(self, loader, evaluate_X=True):
        # returns average closeness score
        if self.genX is None or self.genY is None or self.disX is None or self.disY is None:
            raise Exception('Model cannot be saved as it has not been initialized!')

        name = ''
        self.genX.eval()
        self.genY.eval()
        total_loss = 0.0
        iterations = 0
        for img in loader:
            iterations += 1
            if self.device is not None:
                img = img.cuda()

            output = None
            if evaluate_X:
                output = self.genX(img)
                output = self.genY(output)
                name = 'X'
            else:
                output = self.genY(img)
                output = self.genX(output)
                name = 'Y'

            iteration_loss = self.get_closeness_score(output, img)
            total_loss += iteration_loss.item()

        total_loss = total_loss / (iterations * 1.0)
        print('Average closeness score each batch for {} set is : {}'.format(name, total_loss))
        return total_loss

    def change_params(self, epochs=None, learning_rate=None, leaky_relu=None, betas=None, lamda=None):
        if epochs is not None:
            self.epochs = epochs
            print('Changed the number of epochs to {}!'.format(self.epochs))
        if learning_rate is not None:
            self.lr = learning_rate
            print('Changed the learning rate to {}!'.format(self.lr))
        if leaky_relu is not None:
            self.leaky_relu_threshold = leaky_relu
            print('Changed the threshold for leaky relu to {}!'.format(self.leaky_relu_threshold))
        if betas is not None:
            self.betas = betas
            print('Changed the betas for Adams Optimizer!')
        if betas is not None or learning_rate is not None:
            self.set_all_params(self.epochs, self.lr, self.leaky_relu_threshold, self.lamda, self.betas)

        if lamda is not None:
            self.lamda = lamda
            print('Lamda value has been changed to {}!'.format(self.lamda))

    def set_all_params(self, epochs, lr, leaky_thresh, lamda, beta):
        self.epochs = epochs
        self.lr = lr
        self.leaky_relu_threshold = leaky_thresh
        self.lamda = lamda
        self.betas = beta
        gen_params = list(self.genX.parameters()) + list(self.genY.parameters())

        self.gen_optim = optim.Adam(gen_params, lr=self.lr, betas=self.betas)
        self.disX_optim = optim.Adam(self.disX.parameters(), lr=self.lr, betas=self.betas)
        self.disY_optim = optim.Adam(self.disY.parameters(), lr=self.lr, betas=self.betas)

        self.lr_schedule_disX = self.get_learning_schedule(self.disX_optim, self.lr_policy)
        self.lr_schedule_disY = self.get_learning_schedule(self.disY_optim, self.lr_policy)
        self.lr_schedule_gen = self.get_learning_schedule(self.gen_optim, self.lr_policy)

        print('Model Parameters are:\nEpochs : {}\nLearning rate : {}\nLeaky Relu Threshold : {}\nLamda : {}\nBeta : {}'
              .format(self.epochs, self.lr, self.leaky_relu_threshold, self.lamda, self.betas))

    def save_checkpoint(self, filename, model_type='unet'):
        if self.genX is None or self.disX is None or self.genY is None or self.disY is None:
            raise Exception('The model has not been initialized and hence cannot be saved !')

        filename = '{}/checkpoints/{}.pth'.format(self.base_path, filename)
        save_dict = {'model_type': model_type, 'disX_dict': self.disX.state_dict(), 'genX_dict': self.genX.state_dict(),
                     'disY_dict': self.disY.state_dict(), 'genY_dict': self.genY.state_dict(),
                     'lr': self.lr,
                     'epochs': self.epochs, 'betas': self.betas, 'image_size': self.image_size,
                     'leaky_relu_thresh': self.leaky_relu_threshold, 'lamda': self.lamda, 'base_path': self.base_path,
                     'count': self.count, 'image_format': self.image_format, 'device': self.device,
                     'residual_blocks': self.residual_blocks, 'layer_size': self.layer_size,
                     'lr_policy': self.lr_policy}

        torch.save(save_dict, filename)

        print('The model checkpoint has been saved !')

    def load_checkpoint(self, filename):
        filename = '{}/checkpoints/{}.pth'.format(self.base_path, filename)
        if not pathlib.Path(filename).exists():
            raise Exception('This checkpoint does not exist!')

        save_dict = torch.load(filename)

        self.betas = save_dict['betas']
        self.image_size = save_dict['image_size']
        self.epochs = save_dict['epochs']
        self.leaky_relu_threshold = save_dict['leaky_relu_thresh']
        self.lamda = save_dict['lamda']
        self.lr = save_dict['lr']
        self.base_path = save_dict['base_path']
        self.count = save_dict['count']
        self.image_format = save_dict['image_format']
        self.device = save_dict['device']
        self.residual_blocks = save_dict['residual_blocks']
        self.layer_size = save_dict['layer_size']
        self.lr_policy = save_dict['lr_policy']

        device = self.get_device()
        if device != self.device:
            error_msg = ''
            if self.device is None:
                error_msg = 'The model was trained on CPU and will therefore be continued on CPU only!'
            else:
                error_msg = 'The model was trained on GPU and cannot be loaded on a CPU machine!'
                raise Exception(error_msg)

        self.initialize_model(model_type=save_dict['model_type'], residual_blocks=self.residual_blocks,
                              layer_size=self.layer_size, lr_schedular_options=self.lr_policy)

        self.genX.load_state_dict(save_dict['genX_dict'])
        self.disX.load_state_dict(save_dict['disX_dict'])
        self.genY.load_state_dict(save_dict['genY_dict'])
        self.disY.load_state_dict(save_dict['disY_dict'])

        print('The model checkpoint has been restored!')
