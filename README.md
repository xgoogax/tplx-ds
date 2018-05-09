# tplx-ds
This is a repository for a recruitment task. It is complementary to the report. 
This repository consists of several files - here I will point out the guidelines on how to run specific blocks (parts of the assignment).
1. The paths to datasets and other details related to the project are stored in the file "config.ini". The first thing that needs to be done is to generate a config file, which is a dictionary that stores all of the variables kept in the "config.ini" file. 
To generate it run:
	- python configuration_setup.py --conf='config.ini'
This will create a config.py file which can be imported to other files as a module.
The next step is running the data_handling.py by
	- python data_handling.py
Running this script will create the needed data structure for the further image analysis and classification. 
To extract the CNN codes from a training and a test set and save this representation in a file, run:
	python cnn_codes.py

The Image Analysis (plotting the images, extracting features and training a shallow classifier) is performed in a notebook image_analysis.ipynb.
The visualization of CNN codes and the final training is performed in the file called "pytorch codes visualization.ipynb".
To run the additional experiment run python experiment_2.py.

Additionally, there is one python files that serve as a place for helper functions:
- "image_feature_extraction.py" stores functions for loading the dataset, labels, transforming the images and extracting some of the features
