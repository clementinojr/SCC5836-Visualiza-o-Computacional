import cv2
import numpy as np
import os
import glob
import pandas as pd
from PIL import Image 
import mahotas 
import matplotlib.pylab as plt
import csv
import time
start_time = time.time()

from Features import *



class imageData(object):
    __slots__ = ['name', 
                'feature1', 
                'feature2', 
                'feature3', 
                'feature4', 
                'feature5',
                'feature6',
                'feature7',
                'feature8',
                'feature9',
                'feature10',
                'feature11',
                'feature12',
                'category']

img1 = imageData()
img1.name = 'imagem'



folder_path_covid = "E:\DataSetCovid\COVID-19_Radiography_Dataset\COVID"
folder_path_normais = "E:/DataSetCovid/COVID-19_Radiography_Dataset/Normal"
#folder_path_intersticiais = "E:\DataSetCovid\dataset-img-rx-torax-COVID-anonimizados\PNG_file\Resize299x299\intersticiais-nao-covid"
#folder_path_viral = "/kaggle_dataset/viral_pneumonia"

##Função que redefine a resolução de uma imagem 1 = 100%; 0.5 = 50%
def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)


##Função que realiza a extrações de características de imagens em uma pasta
def ExtractFeatureDataset(path):
    images_path = os.listdir(path)
    data = []
    for n, image in enumerate(images_path):
        print('Extraindo: ', image, ' Category:', os.path.basename(os.path.normpath(path)), ' Quantidade: ', n, '/', len(images_path))
        img = cv2.imread(os.path.join(path, image))
        #resize_ratio = 0.5
        #img = maintain_aspect_ratio_resize(img, width=int(img.shape[1] * resize_ratio))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.max(2)
        mask = np.ones((img.shape[0], img.shape[1]))
        feature_1 = fos(img, mask)[0]
        print('FOS-Extraido, %s Segundos' % round((time.time() - start_time),2) )
        aux_feature = glcm_features(img)
        feature_2 = np.hstack([aux_feature[0], aux_feature[1]])
        print('GLCM-Extraido, %s Segundos' % round((time.time() - start_time),2) )
        #feature_3 = glds_features(img, mask)[0]
        #print('GLDS-Extraido, %s Segundos' % round((time.time() - start_time),2) )
        #feature_4 = ngtdm_features(img, mask, 8)[0]
        #print('NGTDM-Extraido, %s Segundos' % round((time.time() - start_time),2) )
        #feature_5 = sfm_features(img, mask)[0]
       # print('SGM-Extraido, %s Segundos' % round((time.time() - start_time),2) )
        #feature_6 = lte_measures(img, mask, 3)[0]
        # print('LTE-Extraido, %s Segundos' % round((time.time() - start_time),2) )
        # feature_7 = fps(img, mask)[0]
        # print('FPS-Extraido, %s Segundos' % round((time.time() - start_time),2) )
        # feature_8 = lbp_features_no_mask(img, 8, 2)
        #print('LBP-Extraido, %s Segundos' % round((time.time() - start_time),2) )
        feature_9 = lpq_features(img, 7)
        print('LPQ-Extraido, %s Segundos' % round((time.time() - start_time),2) )
        #feature_10 = cv2.HuMoments(cv2.moments(img_gray)).flatten()
        #print('HUMoments-Extraido, %s Segundos' % round((time.time() - start_time),2) )
        #feature_11 = np.hstack([feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7])
        #print('COMBINATION1-Extraido, %s Segundos' % round((time.time() - start_time),2) )
       # feature_12 = np.hstack([feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10])
        #print('COMBINATION2-Extraido, %s Segundos' % round((time.time() - start_time),2) )

        imgData = imageData()
        imgData.name = image
        imgData.feature1 = list(feature_1)
        imgData.feature2 = list(feature_2)
        # imgData.feature3 = list(feature_3)
        # imgData.feature4 = list(feature_4)
        # imgData.feature5 = list(feature_5)
        # imgData.feature6 = list(feature_6)
        # imgData.feature7 = list(feature_7)
        # imgData.feature8 = list(feature_8)
        imgData.feature9 = list(feature_9)
        # imgData.feature10 = list(feature_10)
        # imgData.feature11 = list(feature_11)
        # imgData.feature12 = list(feature_12)
        imgData.category = os.path.basename(os.path.normpath(path))
        data.append(imgData)
    return data

dataCovid = ExtractFeatureDataset(folder_path_covid)
dataNormal = ExtractFeatureDataset(folder_path_normais)
#dataIntersticial = ExtractFeatureDataset(folder_path_intersticiais)
#dataViral = ExtractFeatureDataset(folder_path_viral)

##Combinando para gerar uma lista só para exportarção em CSV
combineData = np.append(dataCovid, dataNormal)
#combineData2 = np.append(combineData, dataIntersticial)
dataSet = combineData

##Titulo de cada atributo
fieldnames = ['Image', 
              'Feature1', 
              'Feature2', 
            #   'Feature3', 
            #   'Feature4', 
            #   'Feature5',
            #   'Feature6',
            #   'Feature7',
            #   'Feature8',
              'Feature9',
            #   'Feature10',
            #   'Feature11',
            #   'Feature12',
              'Category']

##Função que exporta a lista de extrações em CSV 
def WriteCSVFile(path, fieldnames, dataset):
    file = open(path, 'w', newline='', encoding='utf-8')
    writer = csv.writer(file)
    writer.writerow(fieldnames)
    for data in dataset:
        objImg = [data.name, 
                  data.feature1, 
                  data.feature2, 
                #   data.feature3, 
                #   data.feature4, 
                #   data.feature5, 
                #   data.feature6, 
                #   data.feature7, 
                #   data.feature8,
                  data.feature9, 
                #   data.feature10, 
                #   data.feature11, 
                #   data.feature12, 
                  data.category]
        writer.writerow(objImg)

WriteCSVFile('E:\DataSetCovid\COVID-19_Radiography_Dataset\Kdataset_texture_features_raiox.csv',fieldnames, dataSet)