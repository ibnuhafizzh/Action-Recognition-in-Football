
import torch
import torch.nn as nn
import torch.nn.functional as F
# from sklearn.neighbors import NearestNeighbors
import numpy as np



from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import math

class NetVLAD(nn.Module):
    def __init__(self, cluster_size, feature_size, add_batch_norm=True):
        super(NetVLAD, self).__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.clusters = nn.Parameter((1/math.sqrt(feature_size))
                *th.randn(feature_size, cluster_size))
        self.clusters2 = nn.Parameter((1/math.sqrt(feature_size))
                *th.randn(1, feature_size, cluster_size))

        self.add_batch_norm = add_batch_norm
        self.out_dim = cluster_size*feature_size

    def forward(self,x):
        # x [BS, T, D]
        max_sample = x.size()[1]

        # LOUPE
        if self.add_batch_norm: # normalization along feature dimension
            x = F.normalize(x, p=2, dim=2)

        x = x.reshape(-1,self.feature_size)
        assignment = th.matmul(x,self.clusters) 

        assignment = F.softmax(assignment,dim=1)
        assignment = assignment.view(-1, max_sample, self.cluster_size)

        a_sum = th.sum(assignment,-2,keepdim=True)
        a = a_sum*self.clusters2

        assignment = assignment.transpose(1,2)

        x = x.view(-1, max_sample, self.feature_size)
        vlad = th.matmul(assignment, x)
        vlad = vlad.transpose(1,2)
        vlad = vlad - a

        # L2 intra norm
        vlad = F.normalize(vlad)
        
        # flattening + L2 norm
        vlad = vlad.reshape(-1, self.cluster_size*self.feature_size)
        vlad = F.normalize(vlad)

        return vlad

# Menentukan ukuran klaster dan ukuran fitur
cluster_size = 2
feature_size = 3

# Membuat objek NetVLAD
netvlad = NetVLAD(cluster_size, feature_size)

# Membuat masukan sederhana
input_data1 = torch.tensor(
    [[[1.1, 2.2, 3.3], 
      [4.4, 5.5, 6.6]]]
    )  # Dimensi: [BS, T, D]
input_data2 = torch.tensor([[[7.7, 8.8, 9.9], [9.7, 8.6, 7.5]]])  # Dimensi: [BS, T, D]
input_all = torch.cat((input_data1, input_data2), dim=1)
print("frame feature all:\n", input_all)
netvlad_ori = netvlad(input_all)
print("\nOrigina NetVLAD result:\n", netvlad_ori)


print("frame feature before:\n", input_data1)
print("frame feature after:\n", input_data2)

# Melakukan feedforward pada model NetVLAD
before = netvlad(input_data1)
print("\nNetVLAD before: result\n", before)
after = netvlad(input_data2)

# Menampilkan hasil keluaran
print("\nNetVLAD after result:\n", after)

aggregation_result = torch.cat((before, after), dim=1)
print("\nTCA result (aggregation before & after):\n", aggregation_result)

