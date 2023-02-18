#%%
import SimpleITK as sitk
import pandas as pd
import numpy as np
import os
from multiprocessing.pool import Pool, ThreadPool
from radiomics import featureextractor, getTestCase

NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of multiprocessing threads

#%%
extractor = featureextractor.RadiomicsFeatureExtractor('exampleCT.yaml')
features = []
rootPath = 'E:\\Data\\DATA\\Neural\\standard_data\\'
data_df = pd.read_excel(rootPath + 'total_data.xlsx')
idx = data_df.index.to_list()

def extra_features(i):
    if data_df.at[i, 'label'] == 0:
        img_path = os.path.join(rootPath, 'AQP4', 'origin_data', data_df.at[i, 'origin_data'])
        seg_path = os.path.join(rootPath, 'AQP4', 'segment', data_df.at[i, 'segment'])
    else:
        img_path = os.path.join(rootPath, 'MOG', 'origin_data', data_df.at[i, 'origin_data'])
        seg_path = os.path.join(rootPath, 'MOG', 'segment',data_df.at[i, 'segment'])
    print(i,img_path)
    img = sitk.ReadImage(img_path)
    seg = sitk.ReadImage(seg_path)
    result = extractor.execute(img, seg)
    return result

# for i in idx[::-1]:
#     features.append(extra_features(i))
features = list(ThreadPool(NUM_THREADS).imap(extra_features, idx))

#%%
feature_names = list(sorted(filter(lambda k: k.startswith(("original", 'wavelet', 'log')), features[0])))
# print(feature_names, features[0])
samples = np.zeros((len(features),len(feature_names)))
for case_id in range(len(features)):
    a = np.array([])
    for feature_name in feature_names:
        a = np.append(a, features[case_id][feature_name])
    samples[case_id,:] = a
    
# May have NaNs
samples = np.nan_to_num(samples)
print(samples.shape)

#%%
df = pd.DataFrame(data=samples, columns=feature_names,index=None)
df = pd.concat([data_df, df], axis=1)
df.to_excel('./features_total.xlsx',index=None)

