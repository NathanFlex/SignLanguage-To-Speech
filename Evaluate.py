import numpy as np
import torch
from torchvision.models.video.resnet import r2plus1d_18
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from torchmetrics import Accuracy, F1Score

MODELPATH = r"C:\Users\HP\PycharmProjects\VIT\Project\ModelParameters\model.pth"

def load_numpy(save_dir, filename="video_data"):
    data_path = save_dir + "/" + f"{filename}.npy"
    labels_path = save_dir + "/" + f"{filename}_labels.npy"

    data = np.load(data_path)
    labels = np.load(labels_path)

    return data, labels

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert data to tensor
        data_tensor = torch.from_numpy(self.data[idx]).float()
        label_tensor = torch.tensor(self.labels[idx])
        return data_tensor, label_tensor


data,labels = load_numpy(r'C:\Users\HP\PycharmProjects\VIT\Project\VideoData')
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

dataset = CustomDataset(data, encoded_labels)
model = r2plus1d_18(weights = None)
model.fc = nn.Linear(model.fc.in_features, 50)

batch_size = 2
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
params = torch.load(MODELPATH)
model.load_state_dict(params)
model.eval()

accuracy = Accuracy(task='multiclass',num_classes=50)
f1 = F1Score(task='multiclass',num_classes=50, average='macro')

accuracy = accuracy.to(device)
f1 = f1.to(device)

with torch.no_grad():
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output, 1)

        accuracy(predicted, target)
        f1(predicted, target)

print(f"Accuracy: {accuracy.compute()}")
print(f"F1 Score: {f1.compute()}")
