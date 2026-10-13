import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load matrix
with open('c:/Users/conta.LAPTOP-IR41J1UC/Desktop/Projects/CliniCare AI/CliniCare v2/backend/data/model_results.json', 'r') as f:
    data = json.load(f)

cm = np.array(data['confusion_matrix'])
classes = list(data['per_class'].keys())

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes,
            annot_kws={'size': 14})
plt.title('V5 Model Confusion Matrix (OOF)', fontsize=16)
plt.ylabel('Actual Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
plt.tight_layout()

# Save the plot
plt.savefig('c:/Users/conta.LAPTOP-IR41J1UC/Desktop/Projects/CliniCare AI/CliniCare v2/backend/data/confusion_matrix.png', dpi=300)
print('Saved plot to c:/Users/conta.LAPTOP-IR41J1UC/Desktop/Projects/CliniCare AI/CliniCare v2/backend/data/confusion_matrix.png')
