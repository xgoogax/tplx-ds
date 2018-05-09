"""
	This is an additional experiment that was possible to achieve thanks to the access to GPUs. 
	The parameters for the classifier are hard coded based on the Grid Search results presented in pytorch code visualization notebook. 
	The parameter of change is the fraction of the dataset used for training and testing. 
	I am trying to prove a point that if a training set increases, the average accuracy also increases. 
	Of course, it would be ideal if each of the runs was fine-tuned by cross-validation, 
	but I suppose that this script serves as a good starting point for further exploration.  
"""

from config import config
import pickle
import torch
import torch.nn as nn
import numpy as np
from cnn_codes import *
import time
from data_handling import restrict_dataset
from image_feature_extraction import load_images, reshape_images, even_instances
import matplotlib.pyplot as plt 
import os
from torchvision import models, transforms
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
if not torch.cuda.is_available():
	print("I am not going to play a waiting game with you. Get some GPUs to run this experiment.")
	exit()

"""
WARNING

"""
cifarpath = config['paths']['cifardatapath']
path_to_restricted_dataset = config['paths']['restricted_dataset_path']
fractions = [0.05,0.1, 0.25, 0.4, 0.5, 0.75, 0.9]
model_vgg19 = models.vgg19(pretrained=True)
model_vgg19.cuda()
layers = nn.Sequential(*list(model_vgg19.classifier.children())[:-1])
model_vgg19.classifier = layers
gamma=0.1
c=1
scaler = StandardScaler()
svm = SVC(kernel='linear', C=c, gamma=gamma)
accuracy = []
for params in model_vgg19.parameters():
        params.requires_grad = False
if not os.path.exists(cifarpath):
	print("Please run data_handling.py again, but don't forget to modify restrict_dataset function with argument remove_original=False")
else:
	dataset_path_exp2 = path_to_restricted_dataset + "_exp2"
	if not os.path.exists(dataset_path_exp2):
		os.makedirs(dataset_path_exp2)
	for fraction in fractions:
		restrict_dataset(cifarpath, dataset_path_exp2, fraction=fraction, remove_original=False)
		train_data, train_labels, test_data, test_labels = load_images(dataset_path_exp2)
                print(train_data.shape)	  
                train_loader = generate_dataset_and_loader(train_data)
                test_loader = generate_dataset_and_loader(test_data)
		cnn_codes_train = retrieve_cnn_codes(train_loader, model_vgg19, output_shape=(train_data.shape[0], 4096), batch_size=batch_size)
		cnn_codes_test = retrieve_cnn_codes(test_loader, model_vgg19, output_shape=(test_data.shape[0], 4096), batch_size=batch_size)
		scaled_data = scaler.fit_transform(cnn_codes_train)
		scaled_test_data = scaler.transform(cnn_codes_test)
		svm.fit(scaled_data, train_labels)
		preds= svm.predict(scaled_test_data)
		accuracy.append(accuracy_score(test_labels, preds))
		print("Finished for fraction {}".format(fraction))

plt.plot(fractions, accuracy)
plt.scatter(fractions, accuracy)
plt.xlabel("Fraction of dataset used")
plt.ylabel("Average accuracy over classes")
plt.show()
plt.savefig("accuracies.png")

