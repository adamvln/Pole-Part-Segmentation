#
#
#      0=================================0
#      |         Data Partitioning       |
#      0=================================0
#      This code partitions the dataset into three splits: train, test and validation
#      The code partitions the data using multilabel stratified split to make sure all classes exist in all splits
#      The code is prepared for point cloud in .laz format
#      1) Modify the path to the .laz files in line 29
#      2) Modify the 'test_size' variable (what percentage of the dataset is dedicated to validation and test?)
#      3) Modify the 'val_size' variable (what percentage of the splitted test set is dedicated to validation?)
#      The output of the code is three folders of 'train_ply', 'test_ply', and 'val_ply' inside the original data folder
# ----------------------------------------------------------------------------------------------------------------------
#
#      Sara Yousefimashhoor - 07/2022
#
# ----------------------------------------------------------------------------------------------------------------------


from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from plyfile import PlyData
import numpy as np
import pandas as pd
import os
import shutil

# Introduce the path to the .laz files
folder_path = 'KPConv/Data/poles_modified_ply'
wd = os.getcwd()
# Extract the name of the .laz files and store in a list
ply_files = [x for x in os.listdir(folder_path) if x.endswith('.ply')]
print(ply_files)
# Assuming there are 2 possible labels in label_streetlight
exs = np.zeros((len(ply_files), 2), dtype=int)
parts = []
print("ok")
# Iterate over each ply file
for i, ply_file in enumerate(ply_files):
    # Read the PLY file
    cloud_path = os.path.join(folder_path, ply_file)
    cloud = PlyData.read(cloud_path)
    j = 0
    
    # Check if 'vertex' element and 'label_streetlight' property exist
    if 'label_streetlight' in cloud['vertex'].data.dtype.names:
        label_streetlight = cloud['vertex']['label_streetlight']
        # Iterate over the possible labels
        for m in range(2):  # Assuming labels are 0 or 1
            # Check if the label m is present in label_streetlight
            bn = np.any(label_streetlight == m)
            exs[i, m] = int(bn)
            j += int(bn)
            
    parts.append(j)
    print(exs[i])
# Preparing the data to be splited (turn lists to arrays)
X = np.array(ply_files)
y = np.array(exs)
print(max(parts))

test_size = 0.2
# Setting the split parameters
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=36)
# Storing the split result as an array
for train_index, test_index in msss.split(X, y):
    print("TRAIN:", train_index,'\n', 'test1', test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# Name of the chosen files for each split
# print('X_train:\n',X_train, '\n',type(X_test))
# print('X_test:\n',X_test,'\n',type(X_test))

# Clean and Create a directory to save training files
splits = ['train_ply', 'test_ply', 'val_ply']
for split in splits:
    if not os.path.exists(os.path.join(folder_path,split)):
        os.mkdir(os.path.join(folder_path,split))
    # else:
    #     shutil.rmtree(os.path.join(folder_path,split))
# Separate the training files and store it in a separate folder
for name in X_train:
    shutil.copy(os.path.join(folder_path,name), os.path.join(folder_path, 'train_ply'))
# Split the test data into test and validation set
exs2=[]
for name in X_test:
    exs2.append(exs[ply_files.index(name)])
X_test=np.array(X_test)
y2=np.array(exs2)
val_size = 0.5
msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size= val_size, random_state=45)
for test2_index, validation_index in msss2.split(X_test, y2):
    print("TEST:", test2_index,'\n', "Validation:", validation_index)
    Final_test, validation = X_test[test2_index], X_test[validation_index]
    y_test2, y_validation = y2[test2_index], y2[validation_index]
# Move the files into test and validation folders
for name in Final_test:
    shutil.copy(os.path.join(folder_path,name), os.path.join(folder_path, 'test_ply'))
for name in validation:
    shutil.copy(os.path.join(folder_path,name), os.path.join(folder_path, 'val_ply'))
