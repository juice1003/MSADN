Data_Analysis: This folder contains the prediction data of MSADN and its visual analysis, such as confusion matrix, PR&ROC curve drawing, etc.

NCY_Ten_Fold_Data: This folder contains the data of MSADN's ten-fold cross validation on the NCY dataset.

nRC_Ten_Fold_Data: This folder contains the data of MSADN's ten-fold cross validation on the nRC dataset.

MSADNmodel.py: Source code of the MSADN model.

Main_Program_NCY.py & Main_Program_nRC.py: These files are used for data loading and training of the MSADN model.

Performance_Test_NCY.py & Performance_Test_nRC.py:  These files are used to evaluate the performance of the trained MSADN model.

kmer_nRC.py, one_hot_nRC.py, NCPND_NCY.py & NCPNC_nRC.py: These files are three methods for encoding ncRNA sequence data. 

The code runs in the following environment:
Python 3.10
torch 2.0.1
