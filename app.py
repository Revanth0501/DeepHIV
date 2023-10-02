import streamlit as st
import deepchem 
from rdkit import Chem
import torch
import torch_geometric
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as pyg_nn
from rdkit.Chem import rdmolops
from rdkit.Chem import Draw


class GNNModel(nn.Module):

    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GNNModel, self).__init__()
        self.conv1 = pyg_nn.GraphConv(num_node_features, hidden_channels)
        self.conv2 = pyg_nn.GraphConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = pyg_nn.global_mean_pool(x, data.batch)
        x = self.fc(x)
        return torch.sigmoid(x)

def run():

    st.title('HIV INHIBITON ')
    smiles=st.text_input('Please Enter The Molecule')
    model = GNNModel(30,64,2)
    model.load_state_dict(torch.load('D:\Projects\DeepHIV-Predicting-HIV-Inhibition-with-Graph-Neural-Networks-in-PyTorch\model_weights.pth'))
    model.eval()
    if len(smiles)!=0:
        featurizer = deepchem.feat.MolGraphConvFeaturizer(use_edges=True)
        mol = Chem.MolFromSmiles(smiles)
        img=Draw.MolToImage(mol)
        st.image(img)
        data = featurizer.featurize(mol)[0]
        data=data.to_pyg_graph()
        with torch.no_grad():
            output = model(data)
            predicted_class = torch.argmax(output).item()
        if predicted_class==1:
            st.text('YES')
        else:
            st.text('NO')

if __name__=="__main__":
    run()