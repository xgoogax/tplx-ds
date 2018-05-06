import skimage
import numpy as np 
from data_handling import unpickle
import os

def load_images(path):
	train_data = []
	train_labels = []
	test_data = []
	test_labels = []

	if os.path.exists(path) and len(os.listdir(path))>0:
		for file in os.listdir(path):
			if "meta" not in file:
				curr_data = unpickle(os.path.join(path, file))
				if 'data' in file and 'test' not in file:
					data = train_data
					labels = train_labels
				elif 'test' in file:
					data = test_data
					labels = test_labels
				data.extend(curr_data['data'])
				labels.extend(curr_data['labels'])

	return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)

#delete some instances of some labels if their total appearance is higher than a limit
def even_instances(data, labels, limit=500):
	for label in list(np.unique(labels)):
		label_instances = np.where(labels == label)[0]
		if len(label_instances) > limit:
			to_delete = label_instances[limit:]
		#deleting rows
			data = np.delete(data, to_delete, axis=0)
			labels = np.delete(labels, to_delete, axis=0)
	return data, labels

#this is re-usal of some code that I used at one of my classes for loading cifar10 images 
def reshape_images(data, size_image, n_channels):
	data_float = skimage.img_as_float(data)
	rgb_images = np.zeros(data.shape[0]*size_image*size_image*n_channels)
	for i in range(n_channels):
		channel_values = data_float.T[i*size_image*size_image:(i+1)*size_image*size_image]
		rgb_images[i::n_channels] = channel_values.T.ravel()
	return rgb_images.reshape(data.shape[0], size_image, size_image, n_channels)

def extract_mean_color(image):
	return np.mean(image,axis=(0,1))

