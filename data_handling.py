from config import config 
import urllib.request
import numpy as np
from config import config
from bs4 import BeautifulSoup as bs
import requests
import os.path
import tarfile
import pickle
import random 
import math


#specifying some global variables that will be referred to in functions
compressed_file_path = config['paths']['compressed_file']
cifarpath = config['paths']['cifardatapath']
url = config['paths']['dataseturl']
dataset_version = config['dataset_details']['dataset_version']
lang = config['dataset_details']['programming_language']
data_path = config['paths']['datapath']
path_to_restricted_dataset = config['paths']['restricted_dataset_path']


#this function downloads the data from the url parameter and stores it in the folder specified in parent_path parameter
def download_file(url, parent_path=data_path):
	filename = url.split("/")[-1]
	save_path = os.path.join(parent_path, filename)
	if os.path.isfile(save_path):
		print("File already exists")
	else:
		#assuming that data folder already exists - otherwise os.makedirs could be used, but could cause problem with permissions
		try:
			with open(save_path, 'wb') as f:
				url_content = requests.get(url)
				f.write(url_content.content)
			print("Download finished")
		except:
			print("Couldn't download the file")
			print("Please create a folder specified in your parent_path before saving the data")

#this is a long workaround for parsing the CIFAR-10 website and getting the data 

#this function visits the cifar10 websites and tries to find a link with details passed by parameter link_details
#in link details - elements of the list are "keywords" to filter the links - in config ini if you wanted matlab instead of python, you could change the corresponding parameter to matlab and the script would download this version instead of python


def retrieve_link(url, link_details, download=False):
	#read the url 
	with urllib.request.urlopen(url) as website:
		url_content = website.read()
	url_content_structured = bs(url_content, 'html.parser')

	#find all links
	all_links = url_content_structured.find_all("a", href=True)

	#find valid links that contain specific version of CIFAR and programming language - by the text near the tag
	valid_links = list(filter(lambda x: all(detail in x.text.lower().split(" ") for detail in link_details), all_links))
	if len(valid_links) ==0:
		print("No links matching your needs could be found")
		return
	else:
		modified_links = []
		for link in valid_links:
			#unfortunately this is very specific and for this particular website, otherwise I wouldn't split
			current_link = os.path.join("/".join(url.split("/")[:-1]),link.get("href"))

			#you can specify if you want to download it immediately, or first save it somewhere (by default it is not downloaded)
			if download and not os.path.exists(cifarpath):
					print("Attempting to download {}".format(current_link))
					download_file(current_link)
			else:
				modified_links.append(current_link)
		return modified_links


#this function extracts the content of downloaded files. Savepath parameter specifies the path to the compressed file; path specifies path with unpacked files 
def extract_file(path, savepath, remove_compressed=True):
	istar = path.endswith('.tar.gz')
	if not istar:
		print("Sorry, right now I only support unpacking tar.gz files")
	else:
		try:
			with tarfile.open(path, 'r:gz') as tar:
				tar.extractall(path=savepath)
			print("Contents are extracted")
		except:
			print("Couldn't extract contents")
		if remove_compressed:
			remove_compressed_file(path)

#remove compressed file to save some disk space - keep only unpacked files
def remove_compressed_file(path):
	istar = path.endswith('tar.gz')
	if istar:
		os.remove(path)
		print('Removed compressed file at %s', path)


#function from https://www.cs.toronto.edu/~kriz/cifar.html for loading the unpacked data (saved in pickles)
def unpickle(file):
	with open(file, 'rb') as f:
		dict = pickle.load(f, encoding='bytes')
	return dict

"""
here the goal is to create the ultimate data structure for the training set that will be used throughout the experiment
save parameter corresponds to saving it to file for further usage (if not, it will just be returned)
if it is saved, then it will be saved to the restricted dataset - this will have the same structure as the original dataset
this way it can be easily run with different fractions of the entire dataset with very little changes 
"""

#load one batch
def load_dataset(file_path):
	current_data = unpickle(os.path.join(file_path))
	imgs = np.array(current_data[b'data'])
	labels = np.array(current_data[b'labels'])
	return imgs, labels


def remove_original_dataset(path=cifarpath):
	for file in os.listdir(path):
		os.remove(os.path.join(path, file))
	os.rmdir(path)
	print("Removed original dataset at {}".format(path))

def restrict_dataset(initial_dataset_path, restricted_dataset_path, save=True, fraction=config['dataset_details']['fraction'], remove_original=False, random_state=42):
	restricted_dataset = []
	restricted_labels = []
	meta_batches = unpickle(os.path.join(initial_dataset_path, 'batches.meta'))
	class_names =[x.decode('utf-8') for x in meta_batches[b'label_names']]
	mapping_classnames = dict()
	random.seed(random_state)
	for i,value in enumerate(class_names):
		mapping_classnames[i] = value
	#save metafile to the restricted dataset
	if save:
		with open(os.path.join(restricted_dataset_path, 'batches.meta'), 'wb') as f:
			pickle.dump(meta_batches, f)
	if os.path.exists(initial_dataset_path):
		if len(os.listdir(initial_dataset_path))>0:
			for x in os.listdir(initial_dataset_path):
				#unpickling all the batches - extracting fraction from each and then saving it to the restricted_dataset_path
				if '_batch' in x:
					current_subset = {"data": [], "labels": []}
					imgs, labels = load_dataset(os.path.join(initial_dataset_path, x))
					for label in list(mapping_classnames.keys()):
						indices_of_label = np.where(labels == label)[0]
						#now take percent_of_dataset random indices
						number_to_take = math.ceil(len(indices_of_label)*float(fraction))
						random.shuffle(indices_of_label)
						random_subset = indices_of_label[:number_to_take]
						current_subset['data'].extend(imgs[random_subset])
						current_subset['labels'].extend(labels[random_subset])
					current_subset['data'] = np.array(current_subset['data'])
					current_subset['labels'] = np.array(current_subset['labels'])
					if save:
						if not os.path.exists(restricted_dataset_path):
							os.makedirs(restricted_dataset_path)
						with open(os.path.join(restricted_dataset_path, x), 'wb') as f:
							pickle.dump(current_subset, f)
					restricted_dataset.extend(imgs[random_subset])
					restricted_labels.extend(labels[random_subset])
		else:
			print("Empty folder - get some data")
			return 
	else:
		print("No such path {}".format(initial_dataset_path))
		return
	if remove_original:
		remove_original_dataset()
	return restricted_dataset, restricted_labels


if __name__ == "__main__":
	retrieve_link(url, link_details=[dataset_version, lang], download=True)
	if os.path.exists(cifarpath):
		print('Files already extracted')
	else:
		if os.path.exists(compressed_file_path):
			extract_file(compressed_file_path, data_path, remove_compressed=True)
		else:
			print("File could not be extracted, because it could not be found. Please download the data first.")
	restrict_dataset(cifarpath, path_to_restricted_dataset, remove_original=True)
