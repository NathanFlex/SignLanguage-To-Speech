import numpy as np
import torch
from torchvision.models.video.resnet import r2plus1d_18
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

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
        data_tensor = torch.from_numpy(self.data[idx]).float()
        label_tensor = torch.tensor(self.labels[idx])
        return data_tensor, label_tensor


data,labels = load_numpy(r'C:\Users\HP\PycharmProjects\VIT\Project\VideoData')
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

dataset = CustomDataset(data, encoded_labels)

batch_size = 2
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
criterion = nn.CrossEntropyLoss()  # Assuming a classification task
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""model = r2plus1d_18(pretrained = False)
model.fc = nn.Linear(model.fc.in_features, 50)
model = model.to(device)"""

model = r2plus1d_18()
model.fc = nn.Linear(model.fc.in_features, 50)
model = model.to(device)
params = torch.load(MODELPATH)
model.load_state_dict(params)
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

for epoch in range(num_epochs):
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

torch.save(model.state_dict(),MODELPATH)




