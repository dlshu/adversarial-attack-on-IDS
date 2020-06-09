import numpy as np
from glob import glob
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

InD = np.zeros((0,79),dtype=object)
data_file_dir = './MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv'
InD=np.vstack((InD,pd.read_csv(data_file_dir)))

Dt=InD[:,:-1].astype(float)

#Remove nan values
LNMV=InD[~np.isnan(Dt).any(axis=1),-1]
DtNMV=Dt[~np.isnan(Dt).any(axis=1)]
#Remove Inf values
LNMIV=LNMV[~np.isinf(DtNMV).any(axis=1)]
DtNMIV=DtNMV[~np.isinf(DtNMV).any(axis=1)]

del(DtNMV)

np.save('NBx_test', MinMaxScaler().fit_transform(DtNMIV))
np.save('NBy_test', (LNMIV=='BENIGN').astype(int))