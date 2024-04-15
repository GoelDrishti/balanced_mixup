import os
import os.path as osp
from PIL import Image
from torchvision.transforms import Resize
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#Function to load the images with the 3 channels; as required in the codebase 
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

# All the images are stored in the same file, while for each set (train, val and test) an individual csv file is made that stores the imageId, category Class, and the number of instances of each image in the dataset
rsz = Resize((512,640))
rootDirTrain = '/content/chest-xray-pneumonia/chest_xray/train' #data_P/labeled-images/'
outDirTrain = '/content/chest-xray-pneumonia/chest_xray/imgs_new/rs_imgs_PIL' 
os.makedirs(outDirTrain, exist_ok=True)

df_all_mod = pd.DataFrame(columns=['image_id', 'finding_name', 'finding'])
for dirName, subdirList, fileList in os.walk(rootDirTrain):
    print(dirName)
    #print(subdirList)
    #print(fileList)
    for fname in fileList:
        if 'jpeg' in fname:
            img = pil_loader(osp.join(dirName, fname))
            img_res = rsz(img)
            img_res.save(osp.join(outDirTrain,fname))
            subdir_name = dirName.split('/')[-1]
            new_row = {'image_id': fname, 'finding_name': str(subdir_name), 'finding': '-'}
            df_all_mod = pd.concat([df_all_mod, pd.DataFrame([new_row])], ignore_index = True)

findings = df_all_mod['finding_name'].values
findings_list = list(np.unique(findings))
findings_to_class = dict(zip(findings_list, np.arange(len(findings_list))))
print(findings_to_class)
df_all_mod['finding'] = np.where(df_all_mod['finding_name'] == 'NORMAL', 0, 1)

#df_all_mod
df_all_mod.to_csv("/content/chest-xray-pneumonia/chest_xray/imgs_new/train_image-labels_PIL.csv")
###########################################################################################
rootDirVal = '/content/chest-xray-pneumonia/chest_xray/val' #data_P/labeled-images/'
outDirVal = '/content/chest-xray-pneumonia/chest_xray/imgs_new/rs_imgs_PIL' #'data_P/images/'
os.makedirs(outDirVal, exist_ok=True)

df_all_mod = pd.DataFrame(columns=['image_id', 'finding_name', 'finding'])
for dirName, subdirList, fileList in os.walk(rootDirVal):
    print(dirName)
    #print(subdirList)
    #print(fileList)
    for fname in fileList:
        if 'jpeg' in fname:
            img = pil_loader(osp.join(dirName, fname))
            img_res = rsz(img)
            img_res.save(osp.join(outDirVal,fname))
            subdir_name = dirName.split('/')[-1]
            new_row = {'image_id': fname, 'finding_name': str(subdir_name), 'finding': '-'}
            df_all_mod = pd.concat([df_all_mod, pd.DataFrame([new_row])], ignore_index = True)

findings = df_all_mod['finding_name'].values
findings_list = list(np.unique(findings))
findings_to_class = dict(zip(findings_list, np.arange(len(findings_list))))
print(findings_to_class)
df_all_mod['finding'] = np.where(df_all_mod['finding_name'] == 'NORMAL', 0, 1)

#df_all_mod
df_all_mod.to_csv("/content/chest-xray-pneumonia/chest_xray/imgs_new/val_image-labels_PIL.csv")

###########################################################################################

rootDirTest = '/content/chest-xray-pneumonia/chest_xray/test' #data_P/labeled-images/'
outDirTest = '/content/chest-xray-pneumonia/chest_xray/imgs_new/rs_imgs_PIL' #'data_P/images/'
os.makedirs(outDirTest, exist_ok=True)

df_all_mod = pd.DataFrame(columns=['image_id', 'finding_name', 'finding'])
for dirName, subdirList, fileList in os.walk(rootDirTest):
    print(dirName)
    #print(subdirList)
    #print(fileList)
    for fname in fileList:
        if 'jpeg' in fname:
            img = pil_loader(osp.join(dirName, fname))
            img_res = rsz(img)
            img_res.save(osp.join(outDirTest,fname))
            subdir_name = dirName.split('/')[-1]
            new_row = {'image_id': fname, 'finding_name': str(subdir_name), 'finding': '-'}
            df_all_mod = pd.concat([df_all_mod, pd.DataFrame([new_row])], ignore_index = True)

findings = df_all_mod['finding_name'].values
findings_list = list(np.unique(findings))
findings_to_class = dict(zip(findings_list, np.arange(len(findings_list))))
print(findings_to_class)
df_all_mod['finding'] = np.where(df_all_mod['finding_name'] == 'NORMAL', 0, 1)

#df_all_mod
df_all_mod.to_csv("/content/chest-xray-pneumonia/chest_xray/imgs_new/test_image-labels_PIL.csv")
