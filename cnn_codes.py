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

#path to network_trained_representations
path_network_representations = config['paths']['network_representations']
path_to_restrained_dataset = config['paths']['restricted_dataset_path']
dataset_normalized = config['dataset_details']['network_normalized']

#normalization values taken from this thread: https://github.com/kuangliu/pytorch-cifar/issues/19
#although the normalization values for CIFAR-10 from pytorch forum are 0.5, 0.5, 0.5; 0.5, 0.5, 0.5
transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
             (0.247, 0.243, 0.261)),
    ])

if not dataset_normalized:
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor()
    ])

#batch size can be specified in the config.ini file
batch_size=int(config['dataset_details']['batch_size'])

class restrictedCIFAR10(Dataset):
    def __init__(self, data, transform):

        self.data = data
        self.transform = transform
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        image = self.data[idx,:]
        sample = np.swapaxes(image,0,1)

        if self.transform:
            sample = transform(sample.astype(np.uint8))

        return sample

def generate_dataset_and_loader(data, transform=transform, batch_size=batch_size):
    tensor_dataset = restrictedCIFAR10(data, transform)
    data_loader = DataLoader(tensor_dataset, batch_size, num_workers=3)
    return 

def retrieve_cnn_codes(data, model, output_shape=(5000,4096), batch_size=batch_size):
    start = time.time()
    num_batches = math.ceil(output_shape.shape[0]/batch_size)

    #by default all modules are initialized to train mode, and we want it to be fixed
    model.eval()

    dataiter = iter(data)
    cnn_codes = np.zeros(output_shape)
    now = time.time()
    for i,x in enumerate(dataiter):

        #check if the machine has gpu access
        if torch.cuda.is_available():
            x = x.cuda()
            out = model(x).cpu()
        else:
            out = model(x)
        cnn_codes[batch_size*i:batch_size*(i+1)] = out.numpy()
        print("{}/{} batches completed in {}".format(i,num_batches,time.time()-now))
        now = time.time()
    print("Extraction completed in {} seconds".format(time.time()-start))
    return cnn_codes


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = load_images(path_to_restrained_dataset)
    # make sure every class has the same number of instances
    train_data, train_labels = even_instances(train_data, train_labels, limit=500)
    print("The shape of train data: ", train_data.shape)
    n_channels =3
    #length of width and height of each image
    size_image = 32
    train_data = reshape_images(train_data, size_image, n_channels, float=False)
    test_data = reshape_images(test_data, size_image, n_channels, float=False)
    train_dataloader = generate_dataset_and_loader(train_data, transform)
    test_dataloader = generate_dataset_and_loader(test_data, transform)
    print("Data loaded")

    #initialize the pre-trained model
    model_vgg19 = models.vgg19(pretrained=True)
    print("Vgg-19 loaded")
    if torch.cuda.is_available():
        model_vgg19.cuda()
        print('using gpu')
    else:
        print("could not find gpu. using cpu")
#delete the last layer - the dropout layer is left, because it ensures l2-regularization (https://arxiv.org/pdf/1409.1556.pdf)
    layers = nn.Sequential(*list(model_vgg19.children())[:-1])
    model_vgg19.classifier = layers
    print("Vgg19 last layer deleted")
    print(model_vgg19)
    print("Starting feature extraction...")
    for params in model_vgg19.parameters():
        params.requires_grad = False
    cnn_codes_train = retrieve_cnn_codes(train_dataloader, model_vgg19, output_shape=(train_data.shape[0], 4096), batch_size=batch_size)
    cnn_codes_test = retrieve_cnn_codes(test_dataloader, model_vgg19, output_shape=(test_data.shape[0], 4096), batch_size=batch_size)

    cnn_codes = {"train": cnn_codes_train, "test": cnn_codes_test, "train_labels": train_labels, "test_labels": test_labels}
    if not dataset_normalized:
        path_network_representations = path_network_representations + "_not_normalized"
    with open(path_network_representations, 'wb') as f:
        pickle.dump(cnn_codes, f)
    print("CNN codes saved in a file {}".format(path_network_representations))
