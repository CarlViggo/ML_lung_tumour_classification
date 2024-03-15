# Assume model, test_loader, and device are correctly set up as in your script
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
from partially_pretrained_architecture import gen_alexnet, simple_alexnet
from utils import get_train_valid_loader, get_test_loader
import matplotlib.pyplot as plt
import pickle as pkl

# Correct path to your model
model_path = "C:\\Users\\Johan\\Documents\\Tetration\\fold_8_pul_nod_model_FINAL_DICT.pth"

# Load the model
model = torch.load(model_path, map_location="cuda")
device = "cuda"
model.eval()  # Set the model to evaluation mode
# Lists to store predictions and actual labels
all_preds = []
all_labels = []

num_classes = 2
num_epochs = 500
batch_size = 128
learning_rate = 0.001
image_size = 128
seed = 69

train_loader, valid_loader = get_train_valid_loader(
                        data_dir='datasets/pul_nodules/train',
                        batch_size=batch_size,
                        augment=True,
                        random_seed=1, 
                        size = image_size, 
                        imagenet=True)

test_loader = get_test_loader(
                        data_dir='datasets/pul_nodules/test',
                        batch_size=batch_size, 
                        size = image_size,
                        imagenet= True)

model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())  # Save predictions
        all_labels.extend(labels.cpu().numpy())  # Save actual labels

from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix
cm = confusion_matrix(all_labels, all_preds)
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Assuming all_preds and all_labels are your model's predictions and true labels

#RANDOM FOREST METRICS
#cm = np.array([[114,   4],
       #[  8, 114]])

# Calculate TP, TN, FP, FN for sensitivity and specificity
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

# Calculate Sensitivity and Specificity
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

# Plotting with adjusted figure size and layout
fig, ax = plt.subplots(figsize=(8, 8))  # Increased figure size
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax)

# Labels, title, and ticks
label_font = {'size':'14'}
ax.set_xlabel('Predicted labels', fontdict=label_font)
ax.set_ylabel('True labels', fontdict=label_font)
ax.set_title('AlexNet Confusion Matrix', fontdict={'size':'16'})
ax.tick_params(axis='both', which='major', labelsize=12)

# Adjust tick marks and labels
class_names = ['Negative', 'Positive']
ax.set_xticklabels(class_names, ha="center")
ax.set_yticklabels(class_names, va="center")

# Adjust subplot parameters to give specified padding
plt.subplots_adjust(bottom=0.2)  # Adjust the bottom

# Place Sensitivity and Specificity text below the matrix
plt.figtext(0.5, 0.05, f'Sensitivity: {sensitivity:.2f}', ha="center", va="top", fontsize=12)
plt.figtext(0.5, 0.01, f'Specificity: {specificity:.2f}', ha="center", va="top", fontsize=12)

plt.show()
