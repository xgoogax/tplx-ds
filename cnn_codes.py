from config import config
import pickle
import torch
import numpy as np
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import time
from data_handling import unpickle
from image_feature_extraction import load_images, reshape_images, even_instances
import matplotlib.pyplot as plt 
import skimage.transform

#path to network_trained_representations
path_network_representations = config['paths']['network_representations']

#load the data 
#this process can be automated since I am also using it in a different file - #TODO
path_to_restrained_dataset = config['paths']['restricted_dataset_path']
train_data, train_labels, test_data, test_labels = load_images(path_to_restrained_dataset)
# make sure every class has the same number of instances
train_data, train_labels = even_instances(train_data, train_labels, limit=500)

print("The shape of train data: ", train_data.shape)
n_channels =3
#length of width and height of each image
size_image = 32
train_data = reshape_images(train_data, size_image, n_channels, float=False)
test_data = reshape_images(test_data, size_image, n_channels, float=False)


#introdcing the data
#normalization values taken from this thread: https://github.com/kuangliu/pytorch-cifar/issues/19
#no normalization for now
#although the normalization values for CIFAR-10 from pytorch forum are 0.5, 0.5, 0.5; 0.5, 0.5, 0.5
transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), 
        #                      (0.247, 0.243, 0.261))
        
    ])
class restrictedCIFAR10(Dataset):
    def __init__(self, data, transform):

        self.data = data
        self.transform = transform
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        image = self.data[idx,:]
        sample = image

        if self.transform:
            sample = transform(image.astype(np.uint8))

        return sample

tensor_train_dataset = restrictedCIFAR10(train_data, transform)
tensor_test_dataset = restrictedCIFAR10(test_data, transform)
print("Data loaded")

# plt.imshow(np.swapaxes(tensor_train_dataset[0].numpy().T, 0,1))
# plt.show()
#batch size is set to 1, because it does not matter how many go through network at once - it is not training anyway
batch_size=1
train_dataloader = DataLoader(tensor_train_dataset, batch_size=batch_size, num_workers=3)
test_dataloader = DataLoader(tensor_test_dataset, batch_size=batch_size, num_workers=3)

#initialize the pre-trained model
model_vgg19 = models.vgg19(pretrained=True)
print("Vgg-19 loaded")

#delete the last layer - the dropout layer is left, because it ensures l2-regularization (https://arxiv.org/pdf/1409.1556.pdf)
model_vgg19.classifier = model_vgg19.classifier[:-1]
print("Vgg19 last layer deleted")


def retrieve_cnn_codes(data, model, output_shape=(5000,4096), batch_size=1):
    start = time.time()
    dataiter = iter(data)
    cnn_codes = np.zeros(output_shape)
    now = time.time()
    for i,x in enumerate(dataiter):
        current_cnn_codes = model(x)
        cnn_codes[batch_size*i:batch_size*(i+1)] = current_cnn_codes.numpy()
        if i%100==0:
            print(i, " completed in {}".format(time.time()-now))
            now = time.time()
    print("Extraction completed in {} seconds".format(time.time()-start))
    return cnn_codes

for params in model_vgg19.parameters():
    params.requires_grad = False

print(model_vgg19)
print("Starting feature extraction...")
cnn_codes_train = retrieve_cnn_codes(train_dataloader, model_vgg19, output_shape=(train_data.shape[0], 4096), batch_size=batch_size)
cnn_codes_test = retrieve_cnn_codes(test_dataloader, model_vgg19, output_shape=(test_data.shape[0], 4096), batch_size=batch_size)

cnn_codes = {"train": cnn_codes_train, "test": cnn_codes_test, "train_labels": train_labels, "test_labels": test_labels}
with open(path_network_representations+"_not_normalized", 'wb') as f:
    pickle.dump(cnn_codes, f)
print("CNN codes saved in a file {}".format(path_network_representations))