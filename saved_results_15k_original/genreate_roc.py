import numpy as np
import matplotlib.pyplot as plt

tpr = np.load("tpr.npy")
fpr = np.load("fpr.npy")
precision = np.load("precision.npy")
recall = np.load("recall.npy")

print("tpr = ", tpr)
print("fpr = ", fpr)

with open ("9910-step-AUC.txt", "r") as myfile:
    auc=myfile.readlines()

print(auc)

plt.title("RTFM", fontsize = 14)
plt.rcParams["font.family"] = "Times New Roman"
plt.text(0.6,0.7, auc[0], fontsize = 15)
plt.text(0.6, 0.5, "AUC: {:.2f}%".format(float(auc[1])*100), fontsize = 15)

plt.plot(fpr, tpr, marker='.', label='roc_shang')
# axis labels

plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)

# show the plot
plt.show()