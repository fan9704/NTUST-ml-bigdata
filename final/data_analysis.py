# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('datasets'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
mitbih_train = pd.read_csv('datasets/mitbih_train.csv', header=None)
mitbih_test = pd.read_csv('datasets/mitbih_test.csv', header=None)
ptbdb_normal = pd.read_csv('datasets/ptbdb_normal.csv', header=None)
ptbdb_abnormal = pd.read_csv('datasets/ptbdb_abnormal.csv', header=None)

# MIT-BIH class distribution
mitbih_labels = mitbih_train.iloc[:, -1].value_counts()
plt.bar(mitbih_labels.index, mitbih_labels.values)
plt.title("MIT-BIH Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Load MIT-BIH dataset (replace with your actual data loading)
mitbih_train = pd.read_csv('datasets/mitbih_train.csv', header=None)

# Relabel classes: 0=Normal, 1=Abnormal (merge classes 1-4)
mitbih_train['label'] = mitbih_train.iloc[:, -1].apply(lambda x: 0 if x == 0 else 1)

# Get value counts for the new labels
merged_labels = mitbih_train['label'].value_counts().sort_index()

# Plot
plt.figure(figsize=(8, 5))
merged_labels.plot(kind='bar', color=['green', 'red'])
plt.title("MIT-BIH Merged Class Distribution (Normal vs. Abnormal)")
plt.xlabel("Class (0=Normal, 1=Abnormal)")
plt.ylabel("Count")
plt.xticks([0, 1], ['Normal', 'Abnormal'], rotation=0)
plt.show()