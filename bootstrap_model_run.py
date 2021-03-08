# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 11:38:17 2021

@author: a-lerf
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

def load_data():
    '''
    loading dataset from sklearn fetch_openml
    '''
    mnist = fetch_openml('mnist_784', version= 1)
    X = pd.DataFrame(mnist['data'])
    y = pd.DataFrame(mnist['target'])
    
    y.columns = ['label']
    
    df = pd.concat([y, X], axis=1)
    
    return df

def show_image(row,df):
    '''
    display image from row of DataFrame without label data
    
    to do:
    add functionality to show image 
    from either complete df or just df without labels
    '''
    
    img_list = df.loc[row,0:].to_numpy(dtype= 'uint8')
    img_mat = img_list.reshape(28,28)
    plt.imshow(img_mat, cmap='binary')
    plt.axis('off')
    plt.show()

def prep_data():
    '''
    prepare data and return train test split data to use for model training, etc.
    
    includes scaling of each pixel value for increased classification performance
    '''
    df = load_data()
    
    df_train, df_test = train_test_split(df, test_size = 0.2)
    
    #bootstrap training data
    df_train_new = bootstrap_data(df_train)
    df_train = df_train.append(df_train_new, ignore_index = True) 
    
    #Split data in values and labels
    df_train_X = df_train.drop('label', axis = 1)
    df_train_y = df_train['label'].copy()

    df_test_X = df_test.drop('label', axis = 1)
    df_test_y = df_test['label'].copy()
    
    #scale values
    scaler = StandardScaler()
    df_train_X_scaled = scaler.fit_transform(df_train_X.astype(np.float64))
    df_test_X_scaled = scaler.fit_transform(df_test_X.astype(np.float64))
    
    return df_train_X_scaled, df_train_y, df_test_X_scaled, df_test_y

def bootstrap_data(df):
    '''
    takes DataFrame and generates four new images for each index
    
    each image is shifted up, down, left and right by one pixel respectively
    '''
    df = df.sort_index()
    idx = 0
    df_out = pd.DataFrame({}, columns= df.columns)
    
    for i in df.index:
        val_list =  df.loc[i].to_list()
        
        for c in range(4):
            idx += 1
            new_list = []
            label = [val_list[0]]
            if c == 0:
                new_list.extend(val_list[29:])
                new_list.extend(val_list[1:29])
                label.extend(new_list)
            elif c == 1:
                new_list.extend(val_list[757:])
                new_list.extend(val_list[1:757])
                label.extend(new_list)
            elif c == 2:
                for pos in range(28):
                    new_list.extend(val_list[2+pos*28:29+pos*28])
                    new_list.extend([val_list[1+pos*28]])
                
                label.extend(new_list)
            else:
                for pos in range(28):
                    new_list.extend([0])
                    new_list.extend(val_list[1+pos*28:28+pos*28])
                
                label.extend(new_list)
            
            df_out.loc[idx] = label
    
    return df_out

start_time = time.time()

print('Preparing and loading data')

df_train_X_scaled, df_train_y, df_test_X_scaled, df_test_y = prep_data()

print("--- %s seconds ---" % (time.time() - start_time))
print('Training model')

knn_clf = KNeighborsClassifier(n_neighbors=3, weights='distance',n_jobs = -1)
knn_clf.fit(df_train_X_scaled, df_train_y)

print("--- %s seconds ---" % (time.time() - start_time))

filename = 'finalized_model.sav'
pickle.dump(knn_clf, open(filename, 'wb'))

df_test_y_predict = knn_clf.predict(df_test_X_scaled)

print(classification_report(df_test_y, df_test_y_predict))

print("--- %s seconds ---" % (time.time() - start_time))