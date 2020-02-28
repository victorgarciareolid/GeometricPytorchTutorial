import torch
from torch_geometric.data import Data

# Datasets
from torch_geometric.datasets import TUDataset

edge_index = torch.tensor([[0, 1, 2, 3],
                           [1, 2, 3, 0]], dtype=torch.long)

x = torch.tensor([[-1], [1], [-1], [1]], dtype=torch.long)

data = Data(x=x, edge_index=edge_index)

print(data)

dataset = TUDataset(root='/home/victor/TFG/GeometricPytorchTutorial/ENXYMES', name='ENZYMES')

print(dataset)

print("Num Classes: ", dataset.num_classes)
print("Num features: ", dataset.num_features)
print("Num node Features: ", dataset.num_node_features)

print("dataset[0]\n", dataset[0])
print("dataset[1]\n", dataset[1])

print("dataset[1:4]\n", dataset[1:4])

print("Iterating Over Dataset")

for g in dataset[0:3]:
    print(g)
    print("features: ", g.x)

print("Len: ", len(dataset))
print("Shuffle =)")
d0 = dataset[0]
dataset = dataset.shuffle()
d0_shuffle = dataset[0]
print(d0)
print(d0_shuffle)

print("Some Datasets come with the separation of Train/Val/Test (e.g Planetoid)")

# Mini-Batches. Very Important To Train models! :)
'''
    Geometric Pytorch comes with a very interesting Dataloader.
    Batches are treated as following: 
        - The Adjacency matrices are places over the diagonal
        - The feature vectors are concatenated
        - The class vectors are concatenated
'''

from torch_geometric.data import DataLoader

loader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch in loader:
    print("Reminder: batch = total #nodes\n", batch)
    print("batch.batch maps every column with the respective graph of the batch", batch.batch)
    print(batch.x.size())
    break

# Nice, let's train a model using the Cora dataset
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv

from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/home/victor/TFG/GeometricPytorchTutorial/Cora', name='Cora')

# What is planetoid?
data = dataset[0]
train_mask = data.train_mask
print("y: ", data.y[train_mask])
print("x: ", data.x[train_mask])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = GCNConv(in_channels=dataset.num_node_features, out_channels=16)
        self.conv2 = GCNConv(in_channels=16, out_channels=dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
                    #  features + edge_indexes (adjacency)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training) # if model.train() self.training = True
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net()
model = model.to(device)
data = dataset[0].to(device) # here we have Train/Validation/Test
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train() # to activate dropouts

# This will do gradient descent (batch=train)
for epoch in range(200):
    optimizer.zero_grad() # reset gradients
    out = model(data)
    loss  = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    print("Loss: %.2f" % loss.item())

# Let's see how are doing in test
model.eval()
_, pred = model(data).max(dim=1)
correct = float((pred[data.test_mask].eq(data.y[data.test_mask])).sum().item())
total = float(data.test_mask.sum().item())

print("Accuracy: %f" % (correct/total))