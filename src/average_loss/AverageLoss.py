from average_loss.Loss import *
import pickle
import matplotlib.pyplot as plt


class AverageLoss:
    """
        base_path :
        This specifies the base path where losses and plots are saved!

        models :
        The format of this parameter is (No of models, List of names of all these models)
    """

    def __init__(self, base_path, models=(2, ['Generator', 'Discriminator'])):
        self.all_models = [];
        for i in range(models[0]):
            self.all_models.append(Loss());

        self.model_names = models[1]
        self.index = 0
        self.base_path = base_path

    # Assuming order - As passes to models[1]
    def add_loss(self, losses):
        for i in range(len(self.all_models)):
            self.all_models[i].add(losses[i])

    def plot(self):
        for i in range(len(self.all_models)):
            plt.plot(self.all_models[i].get_loss(), label=self.model_names[i])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curve for Unpaired Image to Image Translation')
        plt.legend()
        plt.savefig('{}/loss_curve_{}.png'.format(self.base_path, self.index))
        plt.show()

    def save(self, filename, save_index=None):
        if save_index is None:
            save_index = self.index
        save_dict = {'index': self.index, 'base_path': self.base_path}
        for i in range(len(self.all_models)):
            save_dict[self.model_names[i]] = self.all_models[i]
        save_dict['model_names'] = self.model_names;
        file_path = '{}/{}_{}'.format(self.base_path, filename, save_index)
        with open(file_path, 'wb') as file:
            pickle.dump(save_dict, file)
        print('Losses have been saved!')
        self.index += 1

    def load(self, filename, index):
        filepath = '{}/{}_{}'.format(self.base_path, filename, index)
        with open(filepath, 'rb') as file:
            save_dict = pickle.load(file)
            self.model_names = save_dict['model_names']
            for i in range(len(self.all_models)):
                self.all_models[i] = save_dict[self.model_names[i]]
            self.index = save_dict['index']
            self.base_path = save_dict['base_path']
        print('Loss checkpoint has been restored!')
