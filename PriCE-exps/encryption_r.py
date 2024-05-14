import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA, FastICA
from parse_coordinates import find_coordinates, find_encrpytcoordinates

from functools import reduce
import math
import random


def is_prime(n):
    if n < 2:
        return 'one'
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def prime_factorization(n):
    factors = []
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    # print(factors)
    return factors


'''Encode the original coordinates from the patch names'''
def encode(attripath, destfpath):
    # Get the patchfilename list
    # base_name = f"{fname}_{x_start}_{y_start}.png"
    x_list = []
    y_list = []
    # folderpath = '/home/ubuntu/test/WSI_distributed_service/WSI/data/cnn_ensemble_updated/TCGA-AC-A23H-01Z-00-DX1/test/subfolder1/'
    # attripath = folderpath+f'patchesnames.csv'
    data = pd.read_csv(attripath)
    patch_list = data['PatchName']

    # generate a matrix of data X and Y
    for i, c in enumerate(patch_list): 
        # print(i,c,subfolder_path)
        (x_coordinate, y_coordinate) = find_coordinates(c) # type: ignore
        x_list.append(int(x_coordinate))
        y_list.append(int(y_coordinate))
            
    xencry_list = []
    yencry_list = []
    X_centered = 0
    Y_centered = 0
  
    # print(len(x_list))

    flag = is_prime(len(x_list))
    factors = prime_factorization(len(x_list)) #1,3 
    # print(flag)
    
    if flag=='one':
        rx = random.randint(1, 1000)
        ry = random.randint(1, 1000)

        xencry_list = x_list
        yencry_list = y_list
        
        encry_patch_names = [f'_{str(j+ xencry_list[j] + rx)}_{str(j + yencry_list[j] + ry)}.png' for j in range(len(xencry_list))]
        df = pd.DataFrame(encry_patch_names)
        df.to_csv(destfpath, header = ['EncryPatchName'])

        return xencry_list, yencry_list, means, X_centered, Y_centered, x_list, y_list, renewX, renewY
    elif flag==False: 
        half = math.ceil(len(factors)/2)
        # half = 1
        first_half, second_half = factors[:half], factors[half:]
        a = reduce(lambda x, y: x * y, first_half)  
        b = reduce(lambda x, y: x * y, second_half)  
        # print(len(x_list), flag, factors, a, b)
        # x_list = y_list
        A = np.array(x_list).reshape((a, b))
        B = np.array(y_list).reshape((a, b))
        # print('Orignal: \n',A, B)


        means = [np.mean(A, axis=0), np.mean(B, axis=0)]
        X_centered = A - np.mean(A, axis=0)
        Y_centered = B - np.mean(B, axis=0)

        # compute the principal components of the centered data using PCA
        Xpca = PCA()
        Ypca = PCA()
        Xpca.fit(X_centered)
        Ypca.fit(Y_centered)

        # transform the data to the new coordinate system defined by the principal components
        X_transformed = Xpca.transform(X_centered)
        Y_transformed = Ypca.transform(Y_centered)
        # print("Transformed data:\n", X_transformed.shape, Y_transformed.shape)
        
        X1 = X_transformed.tolist()
        Y1 = Y_transformed.tolist()
        # print("Transformed data:\n", X1, Y1)

        xencry_list = [X1[i][j] for i in range(len(X1)) for j in range(len(X1[0]))]
        yencry_list = [Y1[i][j] for i in range(len(Y1)) for j in range(len(Y1[0]))]

        Rxencry_list = xencry_list + [0] * (len(x_list) - len(xencry_list))
        Ryencry_list = yencry_list + [0] * (len(y_list) - len(yencry_list))

        rx = random.randint(1, 1000)
        ry = random.randint(1, 1000)
        
        renewX = [j + Rxencry_list[j] + rx for j in range(len(Ryencry_list))]
        renewY = [j + Ryencry_list[j] + ry for j in range(len(Ryencry_list))]

        encry_patch_names = [f'_{str(j+ Rxencry_list[j] + rx)}_{str(j + Ryencry_list[j] + ry)}.png' for j in range(len(Ryencry_list))]
        df = pd.DataFrame(encry_patch_names)
        df.to_csv(destfpath, header = ['EncryPatchName'])

        return xencry_list, yencry_list, means, X_centered, Y_centered, x_list, y_list, renewX, renewY
    elif flag==True: 
        # print({len(x_list)}, " is prime and factors are", factors)
        a, b  = 1, factors[0]
        A = np.array(x_list).reshape((a, b))
        B = np.array(y_list).reshape((a, b))
        # print('Orignal: \n',A, B)

        means = [np.mean(A, axis=0), np.mean(B, axis=0)]
        X_centered = A - np.mean(A, axis=0)
        Y_centered = B - np.mean(B, axis=0)

        # compute the principal components of the centered data using PCA
        Xpca = PCA()
        Ypca = PCA()
        Xpca.fit(X_centered)
        Ypca.fit(Y_centered)

        # transform the data to the new coordinate system defined by the principal components
        X_transformed = Xpca.transform(X_centered)
        Y_transformed = Ypca.transform(Y_centered)
        # print("Transformed data:\n", X_transformed.shape, Y_transformed.shape)
        
        X1 = X_transformed.tolist()
        Y1 = Y_transformed.tolist()
        # print("Transformed data:\n", X1, Y1)

        xencry_list = [X1[i][j] for i in range(len(X1)) for j in range(len(X1[0]))]
        yencry_list = [Y1[i][j] for i in range(len(Y1)) for j in range(len(Y1[0]))]

        Rxencry_list = xencry_list + [0] * (len(x_list) - len(xencry_list))
        Ryencry_list = yencry_list + [0] * (len(y_list) - len(yencry_list))

        rx = random.randint(1, 1000)
        ry = random.randint(1, 1000)
        renewX = [j + Rxencry_list[j] + rx for j in range(len(Ryencry_list))]
        renewY = [j + Ryencry_list[j] + ry for j in range(len(Ryencry_list))]

        encry_patch_names = [f'_{str(j+ Rxencry_list[j] + rx)}_{str(j + Ryencry_list[j] + ry)}.png' for j in range(len(Ryencry_list))]
        df = pd.DataFrame(encry_patch_names)
        df.to_csv(destfpath, header = ['EncryPatchName'])

        return xencry_list, yencry_list, means, X_centered, Y_centered, x_list, y_list, renewX, renewY






'''Decode the coordinates from the encoded .csv'''
def decode(encodedfile, means, X_centered, Y_centered, destfpath): 
    xencry_list = []
    yencry_list = []

    data = pd.read_csv(encodedfile)
    encryptpatch_list = data['EncryPatchName']

    # parse encrypted coordinates from the encode_coordinates1.csv
    for i, c in enumerate(encryptpatch_list): 
        # print(i,c,subfolder_path)
        x_coordinate, y_coordinate = find_encrpytcoordinates(c) # type: ignore
        # print(x_coordinate, y_coordinate)
        xencry_list.append(float(x_coordinate))
        yencry_list.append(float(y_coordinate))

    flag = is_prime(len(xencry_list))
    factors = prime_factorization(len(xencry_list))
    
    if flag=='one':


        xdecry_list = xencry_list
        ydecry_list = yencry_list

        # Rxdecry_list = xencry_list + [0] * (len(x_list) - len(xencry_list))
        # Rydecry_list = yencry_list + [0] * (len(y_list) - len(yencry_list))

        decry_patch_names = [f'wsi_{str(x_coord)}_{str(y_coord)}.png' for x_coord, y_coord in zip(xdecry_list, ydecry_list)]
        
        df = pd.DataFrame(decry_patch_names)
        df.to_csv(destfpath, header = ['DecryPatchName'])
        return xdecry_list, ydecry_list

    elif flag==False: 
        half = math.ceil(len(factors)/2)
        # half = 1
        first_half, second_half = factors[:half], factors[half:]
        a = reduce(lambda x, y: x * y, first_half)  
        b = reduce(lambda x, y: x * y, second_half)  
        # print(len(x_list), flag, factors, a, b)
        # x_list = y_list

        A = np.array(xencry_list).reshape((a, b))
        B = np.array(yencry_list).reshape((a, b))
        # print('Orignal: \n',A, B)
        # compute the principal components of the centered data using PCA
        pcaX = PCA()
        pcaX.fit(X_centered)
        pcaY = PCA()
        pcaY.fit(Y_centered)

        # transform the data to the new coordinate system defined by the principal components
        X_transformed = pcaX.transform(X_centered)
        Y_transformed = pcaY.transform(Y_centered)
        # print("Transformed data:\n", X_transformed, Y_transformed, X_transformed.shape, Y_transformed.shape)

        # reconstruct the original data from the transformed data
        X_reconstructed = pcaX.inverse_transform(X_transformed)
        Y_reconstructed = pcaY.inverse_transform(Y_transformed)

        # print("Reconstructed data:\n", X_reconstructed, Y_reconstructed)
        
        # # print the original and reconstructed data 
        # print('Original data:\n', X)
        finalX = X_reconstructed + means[0]
        finalY = Y_reconstructed + means[1]

        # finalXica = X_ica_reconstruct 
        # finalYica = Y_ica_reconstruct

        # print('Reconstructed data:\n', finalX,finalY, finalX.shape)
        X1 = finalX.tolist()
        Y1 = finalY.tolist()

        # Xica1 = finalXica.tolist()
        # Yica1 = finalYica.tolist()
        # print("Reconstructed data:\n", X1, Y1)

        xdecry_list = [round(X1[i][j]) for i in range(len(X1)) for j in range(len(X1[0]))]
        ydecry_list = [round(Y1[i][j]) for i in range(len(Y1)) for j in range(len(Y1[0]))]

        # icaxdecry_list = [round(Xica1[i][j]) for i in range(len(Xica1)) for j in range(len(Xica1[0]))]
        # icaydecry_list = [round(Yica1[i][j]) for i in range(len(Yica1)) for j in range(len(Yica1[0]))]

        # Rxdecry_list = xencry_list + [0] * (len(x_list) - len(xencry_list))
        # Rydecry_list = yencry_list + [0] * (len(y_list) - len(yencry_list))

        decry_patch_names = [f'wsi_{str(x_coord)}_{str(y_coord)}.png' for x_coord, y_coord in zip(xdecry_list, ydecry_list)]
            

        df = pd.DataFrame(decry_patch_names)
        df.to_csv(destfpath, header = ['DecryPatchName'])

        return xdecry_list, ydecry_list
    
    elif flag==True: 
        # print({len(xencry_list)}, " is prime and factors are", factors)
        a, b  = 1, factors[0]
        A = np.array(xencry_list).reshape((a, b))
        B = np.array(xencry_list).reshape((a, b))
        # print('Orignal: \n',A, B)
        pcaX = PCA()
        pcaX.fit(X_centered)
        pcaY = PCA()
        pcaY.fit(Y_centered)

        # transform the data to the new coordinate system defined by the principal components
        X_transformed = pcaX.transform(X_centered)
        Y_transformed = pcaY.transform(Y_centered)
        # print("Transformed data:\n", X_transformed, Y_transformed, X_transformed.shape, Y_transformed.shape)

        # reconstruct the original data from the transformed data
        X_reconstructed = pcaX.inverse_transform(X_transformed)
        Y_reconstructed = pcaY.inverse_transform(Y_transformed)

        # ica = FastICA()
        # S_ica_ = ica.fit(A).transform(A)  # Estimate the sources

        # print("Reconstructed data:\n", X_reconstructed, Y_reconstructed, S_ica_)
        
        # # print the original and reconstructed data 
        # print('Original data:\n', X)
        finalX = X_reconstructed + means[0]
        finalY = Y_reconstructed + means[1]

        # print('Reconstructed data:\n', finalX,finalY, finalX.shape)
        X1 = finalX.tolist()
        Y1 = finalY.tolist()
        # print("Reconstructed data:\n", X1, Y1)

        xdecry_list = [round(X1[i][j]) for i in range(len(X1)) for j in range(len(X1[0]))]
        ydecry_list = [round(Y1[i][j]) for i in range(len(Y1)) for j in range(len(Y1[0]))]

        # Rxdecry_list = xencry_list + [0] * (len(x_list) - len(xencry_list))
        # Rydecry_list = yencry_list + [0] * (len(y_list) - len(yencry_list))

        decry_patch_names = [f'wsi_{str(x_coord)}_{str(y_coord)}.png' for x_coord, y_coord in zip(xdecry_list, ydecry_list)]
            

        df = pd.DataFrame(decry_patch_names)
        df.to_csv(destfpath, header = ['DecryPatchName'])

        return xdecry_list, ydecry_list

    
## rename the patches with encode_coordinates.
# orig_data = pd.read_csv(subfolder+'patchesnames.csv')
# OrgPatchName = orig_data['PatchName']

# encode_data = pd.read_csv(encodefpath)
# EncryPatchName = encode_data['EncryPatchName']
            
# # build a dictionary
# dict_coords = {}
# outfile = subfolder+'dictionary.json'
# for ecry, orig in zip(EncryPatchName, OrgPatchName):
#     dict_coords[ecry] = orig      
# with open("sample.json", "w") as outfile:
#     json.dump(dict_coords, outfile)