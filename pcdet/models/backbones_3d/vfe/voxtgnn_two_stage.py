import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate
import torch_scatter

import math
# import spconv
import spconv.pytorch as spconv
from torch_geometric.nn import PointGNNConv

from torch_cluster import knn_graph


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class VoxtGNN_Two_Stage(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, grid_size, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.num_latents = self.model_cfg.NUM_LATENTS
        self.input_dim = self.model_cfg.INPUT_DIM
        self.output_dim = self.model_cfg.OUTPUT_DIM
        self.k_gnn = self.model_cfg.K_GNN
        self.gnn_layers = self.model_cfg.GNN_LAYERS

        self.input_embed = MLP(num_point_features, 16, self.input_dim, 2)
       
        self.pe0 = PositionalEncodingFourier(64, self.input_dim)
        self.pe1 = PositionalEncodingFourier(64, self.input_dim * 2)
        self.pe2 = PositionalEncodingFourier(64, self.input_dim * 4)
        self.pe3 = PositionalEncodingFourier(64, self.input_dim * 8)
    
        self.mlp_vsa_layer_0 = MLP_VSA_Layer(self.input_dim * 1, self.num_latents[0],self.k_gnn)
        self.mlp_vsa_layer_1 = MLP_VSA_Layer(self.input_dim * 2, self.num_latents[1],self.k_gnn)
        self.mlp_vsa_layer_2 = MLP_VSA_Layer(self.input_dim * 4, self.num_latents[2],self.k_gnn)
        self.mlp_vsa_layer_3 = MLP_VSA_Layer(self.input_dim * 8, self.num_latents[3],self.k_gnn)


        self.post_mlp = nn.Sequential(
            nn.Linear(self.input_dim * 16, self.output_dim),
            nn.BatchNorm1d(self.output_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim, eps=1e-3, momentum=0.01)
        )
        
       
        self.register_buffer('point_cloud_range', torch.FloatTensor(point_cloud_range).view(1, -1))
        self.register_buffer('voxel_size', torch.FloatTensor(voxel_size).view(1, -1))
        self.grid_size = grid_size.tolist()
        
        a, b, c = voxel_size
        self.register_buffer('voxel_size_02x', torch.FloatTensor([a * 2, b * 2, c]).view(1, -1))
        self.register_buffer('voxel_size_04x', torch.FloatTensor([a * 4, b * 4, c]).view(1, -1))
        self.register_buffer('voxel_size_08x', torch.FloatTensor([a * 8, b * 8, c]).view(1, -1))

        a, b, c = grid_size
        self.grid_size_02x = [a // 2, b //  2, c]
        self.grid_size_04x = [a // 4, b //  4, c]
        self.grid_size_08x = [a // 8, b //  8, c]
        
    def get_output_feature_dim(self):
        return self.output_dim

    
    def forward(self, batch_dict, **kwargs):

        points = batch_dict['points'] # (n,5),5包括batch、x、y、z、i
        points_offsets = points[:, 1:4] - self.point_cloud_range[:, :3]
        
        
        coords01x = points[:, :4].clone()
        coords01x[:, 1:4] = points_offsets // self.voxel_size
        # pe_raw:(n,3)原始位置
        pe_raw = (points_offsets - coords01x[:, 1:4] * self.voxel_size  ) / self.voxel_size
        coords01x, inverse01x = torch.unique(coords01x, return_inverse=True, dim=0)

        coords02x = points[:, :4].clone()
        coords02x[:, 1:4] = points_offsets // self.voxel_size_02x
        coords02x, inverse02x = torch.unique(coords02x, return_inverse=True, dim=0)
      
        coords04x = points[:, :4].clone()
        coords04x[:, 1:4] = points_offsets // self.voxel_size_04x
        coords04x, inverse04x = torch.unique(coords04x, return_inverse=True, dim=0)

        coords08x = points[:, :4].clone()
        coords08x[:, 1:4] = points_offsets // self.voxel_size_08x
        coords08x, inverse08x = torch.unique(coords08x, return_inverse=True, dim=0)


        src = self.input_embed(points[:, 1:]) # (n,16)

        src = src + self.pe0(pe_raw) # (n,16)
        src = self.mlp_vsa_layer_0(src, inverse01x, coords01x, self.grid_size)

        src = src + self.pe1(pe_raw)
        src = self.mlp_vsa_layer_1(src, inverse02x, coords02x, self.grid_size_02x)

        src = src + self.pe2(pe_raw)
        src = self.mlp_vsa_layer_2(src, inverse04x, coords04x, self.grid_size_04x)

        src = src + self.pe3(pe_raw)
        src = self.mlp_vsa_layer_3(src, inverse08x, coords08x, self.grid_size_08x)

        src = self.post_mlp(src)

        batch_dict['point_features'] = F.relu(src)
        batch_dict['point_coords'] = points[:, :4]
        
        batch_dict['voxel_features'] = F.relu(torch_scatter.scatter_max(src, inverse01x, dim=0)[0])
        batch_dict['voxel_coords'] = coords01x[:, [0, 3, 2, 1]]
        
        return batch_dict

    
# 定义多层感知器
class MyMLP(nn.Module):
    def __init__(self, Ks=[7, 32, 64, 128]):
        '''
            定义的普通多层感知器，Ks[0]是输入特征维度，Ks[len(Ks)-1]是输出的特征维度，中间的是隐藏层特征维度
        '''
        super(MyMLP, self).__init__()
        linears = []
        for i in range(1, len(Ks)):
            linears += [
            nn.Linear(Ks[i-1], Ks[i]),
            nn.ReLU(),
            nn.BatchNorm1d(Ks[i])]
        self.Sequential = nn.Sequential(*linears)
    def forward(self, x):
        
        out = self.Sequential(x)
        return out

class GNN_FFN(nn.Module):
    def __init__(self, auto_offset_MLP_depth_list=[128, 64, 3],
                  edge_MLP_depth_list=[128+3, 128, 128], update_MLP_depth_list=[128, 128, 128], graph_net_layers=3):
        '''
            auto_offset_MLP_depth_list=[128, 64, 3]：自动配准中的输入输出特征维度，对应mlp_h
            edge_MLP_depth_list=[128+3, 128, 128]：边聚合中的输入输出特征维度，128+3中的3是指每次进行逐边特征提取时，需要加入自动配准的三维坐标偏差，对应mlp_f
            update_MLP_depth_list=[128, 128, 128]：节点更新中的输入输出特征维度，对应mlp_g
        '''
        super(GNN_FFN, self).__init__()
        
        self.graph_nets = nn.ModuleList()
        for i in range(graph_net_layers):# 使用几层图网络，也即graph_net_layers，每层的MLP不共享参数
            # torch_geometric.nn中的PointGNNConv，已经自动进行了中心配准offset（也即在message函数中进行）、
            # 在PointGNNConv的forward函数中已经加入残差 
            # PointGNNConv聚合特征默认采用最大池化
            # 自动配准的MLP
            mlp_h = MyMLP(auto_offset_MLP_depth_list)
            # 逐边特征提取MLP
            mlp_f = MyMLP(edge_MLP_depth_list)
            # 逐点特征聚合
            mlp_g = MyMLP(update_MLP_depth_list)
            self.graph_nets.append(PointGNNConv(mlp_h=mlp_h,mlp_f=mlp_f,mlp_g=mlp_g))
        
    def forward(self, x,pos,edge_index):
        '''
            处理的是非空体素
            x->(N,M), N 为体素个数，M是每个体素的特征维度,本文M=128
            pos->(N,3)，N 为体素个数，3是每个体素的3维坐标
            edge_index->每个体素作为一个点构造的图结构
        '''
        # 使用PointGNN网络提取特征每个点的特征（逐体素-逐点进行）
        for k, graph_net in enumerate(self.graph_nets):
            x = graph_net(x, pos, edge_index)              
        return x

class MLP_VSA_Layer(nn.Module):
    def __init__(self, dim, n_latents=8,k_gnn=6,gnn_layers=3):
        super(MLP_VSA_Layer, self).__init__()
        self.dim = dim
        self.k = n_latents 
        self.k_gnn = k_gnn 
        self.gnn_layers = gnn_layers 
        self.pre_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim,eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim,eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim,eps=1e-3, momentum=0.01),
        )

        # the learnable latent codes can be obsorbed by the linear projection
        self.score = nn.Linear(dim, n_latents)

        

        conv_dim = dim * self.k
        self.conv_dim = conv_dim

        self.gnn_ffn = GNN_FFN(auto_offset_MLP_depth_list=[conv_dim, 64, 3],
                  edge_MLP_depth_list=[conv_dim+3, conv_dim, conv_dim], update_MLP_depth_list=[conv_dim, conv_dim, conv_dim], 
                  graph_net_layers=gnn_layers) 
        # conv ffn
        # self.conv_ffn = nn.Sequential(           
        #     nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, groups=conv_dim, bias=False), 
        #     nn.BatchNorm2d(conv_dim),
        #     nn.ReLU(),
        #     nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, groups=conv_dim, bias=False), 
        #     nn.BatchNorm2d(conv_dim),
        #     nn.ReLU(),
        #     # nn.Conv2d(conv_dim, conv_dim, 3, 1, dilation=2, padding=2, groups=conv_dim, bias=False),
        #     # nn.BatchNorm2d(conv_dim),
        #     # nn.ReLU(), 
        #     nn.Conv2d(conv_dim, conv_dim, 1, 1, bias=False), 
        #  ) 
        
        # decoder
        self.norm = nn.BatchNorm1d(dim,eps=1e-3, momentum=0.01)
        self.mhsa = nn.MultiheadAttention(dim, num_heads=1, batch_first=True) 
        
    def forward(self, inp, inverse, coords, bev_shape):
        '''
            self.dim = 16 # 默认特征纬度
            inp:(n,self.dim), eg.(83682,16),输入的特征张量，n为点的数量，16为特征维度
            coords:(n',4), eg.(13842,4),n'体素个数，4体素坐标,点对应的体素坐标（去除了重复坐标（也即唯一体素坐标），很多点可以属于统一体素）
            inverse:(n), eg.(83682),与inp中的n一一对应，也即原始索引，而inverse中每个元素的直对应的是coords行索引，可以存储重复的行索引
                    包含唯一体素坐标在原始有重复的体素坐标张量中的索引，也可以一一对应到每个点
            bev_shape: 数组[W,H,D]对应xyz, eg.[216, 248, 1] , 格网大小，通过点云范围和体素大小计算
        '''
        
        x = self.pre_mlp(inp) #(n,self.dim),eg.(83682,16)

        # encoder
        # self.score(x):(n,self.k) ,eg.(83682,8); 
        attn = torch_scatter.scatter_softmax(self.score(x), inverse, dim=0) # (n,self.k) eg.(83682,8)，计算隐藏码
        # attn[:, :, None]: (n,self.k,1),eg.(83682,8,1); x.view(-1, 1, self.dim): (n,1,self.dim),eg.(83682,1,16)
        # (attn[:, :, None] * x.view(-1, 1, self.dim)): [n, self.k, self.dim] eg.[80957, 8, 16]
        dot = (attn[:, :, None] * x.view(-1, 1, self.dim)).view(-1, self.dim*self.k) #(n,self.dim*self.k),eg.(n,128) x与注意力值乘积
        x_ = torch_scatter.scatter_sum(dot, inverse, dim=0)#(n',self.dim*self.k), eg.(13842, 128) n'体素个数，每个体素中点特征之和，也即体素局部特征呢个

        edge_index = knn_graph(coords[:,1:].contiguous(), k=self.k_gnn, batch=coords[:,0].contiguous(), loop=False)
        
        # 体素间信息交换 
        
        h = self.gnn_ffn(F.relu(x_),coords[:,1:],edge_index)
        h = h[inverse, :] # (n,conv_dim),eg.[83682, 128]

        # conv ffn
        # batch_size = int(coords[:, 0].max() + 1)
        # h = spconv.SparseConvTensor(F.relu(x_), coords.int(), bev_shape, batch_size).dense().squeeze(-1)
        # h = self.conv_ffn(h).permute(0,2,3,1).contiguous().view(-1, self.conv_dim)
        # flatten_indices = coords[:, 0] * bev_shape[0] * bev_shape[1] + coords[:, 1] * bev_shape[1] + coords[:, 2]
        # h = h[flatten_indices.long(), :] 
        # h = h[inverse, :]
       
        # decoder
        hs = self.norm(h.view(-1,  self.dim)).view(-1, self.k, self.dim)
        hs = self.mhsa(x.view(-1, 1, self.dim), hs, hs)[0]
        hs = hs.view(-1, self.dim)
        
        
        # skip connection
        return torch.cat([inp, hs], dim=-1)
        
       

class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """

    def __init__(self, hidden_dim=64, dim=128, temperature=10000):
        super().__init__()
        self.token_projection = nn.Linear(hidden_dim * 3, dim)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        

    def forward(self, pos_embed, max_len=(1, 1, 1)):
        z_embed, y_embed, x_embed = pos_embed.chunk(3, 1)
        z_max, y_max, x_max = max_len
        
        eps = 1e-6
        z_embed = z_embed / (z_max + eps) * self.scale
        y_embed = y_embed / (y_max + eps) * self.scale
        x_embed = x_embed / (x_max + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=pos_embed.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed / dim_t
        pos_y = y_embed / dim_t
        pos_z = z_embed / dim_t

        pos_x = torch.stack((pos_x[:, 0::2].sin(),
                             pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(),
                             pos_y[:, 1::2].cos()), dim=2).flatten(1)
        pos_z = torch.stack((pos_z[:, 0::2].sin(),
                             pos_z[:, 1::2].cos()), dim=2).flatten(1)

        pos = torch.cat((pos_z, pos_y, pos_x), dim=1)

        pos = self.token_projection(pos)
        return pos   
