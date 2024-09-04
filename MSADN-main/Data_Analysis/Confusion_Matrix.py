import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

file_paths = [f'H:/Google download/MSADN/MSADN-main/Data_Analysis/each nRC_Test_CSV/Test{i}_nRC.csv' for i in range(10)]

all_true_labels = []
all_predicted_labels = []

for file_path in file_paths:
    data = pd.read_csv(file_path)
    true_labels = data['Real Label']
    predicted_labels = data['Predict Label']
    all_true_labels.extend(true_labels)
    all_predicted_labels.extend(predicted_labels)

cm = confusion_matrix(all_true_labels, all_predicted_labels)
cm_prob = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

labels = [
    '5S-rRNA', '5-8S-rRNA', 'tRNA', 'Ribozyme', 'CD-box',
    'miRNA', 'Intron-gp-I', 'Intron-gp-II', 'HACA-box',
    'Riboswitch', 'IRES', 'Leader', 'scaRNA'
]


annot = np.empty_like(cm_prob, dtype=object)
for i in range(cm_prob.shape[0]):
    for j in range(cm_prob.shape[1]):
        if cm_prob[i, j] == 0:
            annot[i, j] = '0'
        else:
            annot[i, j] = f'{cm_prob[i, j]:.3f}'


plt.figure(figsize=(15, 15))
sns.heatmap(cm_prob, annot=annot, fmt='', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False,
            annot_kws={"size": 14})
plt.xlabel('Predicted label', size=30)
plt.ylabel('True label', size=30)
plt.xticks(rotation=90, fontsize=20)
plt.yticks(rotation=0, fontsize=20)
plt.title('MSADN', size=36)
plt.subplots_adjust(bottom=0.15, top=0.95, left=0.15, right=0.95)

output_image_path = 'H:/Google download/MSADN/MSADN-main/Data_Analysis/confusion_matrix_nRC/MSADN_confusion_matrix_nRC.png'
plt.savefig(output_image_path)
print(f"Confusion matrix saved to {output_image_path}")
plt.show()