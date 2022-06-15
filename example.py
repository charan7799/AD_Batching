import torch.utils.data as data
import numpy as np
from utils import process_feat
import torch
from torch.utils.data import DataLoader
from kmeans_pytorch import kmeans
import option
import random
torch.set_default_tensor_type('torch.cuda.FloatTensor')
# np.random.seed(7)

def cluster_shang(num_clusters, rgb_list_file): 
    
    f = []
    for i in range(len(rgb_list_file)):
        feature = get_features(rgb_list_file, i)
        f.append(feature.flatten())
        nf = np.array(f)
    x = torch.from_numpy(nf)
    cluster_ids_x, cluster_centers = kmeans(
    X=x, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0')
    )
    return cluster_ids_x, cluster_centers

def label_data_shanghai(rgb_list_file):

    l = [1 for i in range(len(rgb_list_file))]
    l_n =[0 for i in range(175)]
    l[63:] = l_n
    print("labels are ",l)
    return l

def label_data_ucf(rgb_list_file):

    l = [1 for i in range(len(rgb_list_file))]
    l_n =[0 for i in range(810)]
    l[810:] = l_n
    print("labels are ",l)
    return l

def get_features(file_list, index):

    features = np.load(file_list[index].strip('\n'), allow_pickle=True)
    features = np.array(features, dtype=np.float32)
    features = features.transpose(1, 0, 2)  # [10, B, T, F]
    divided_features = []
    for feature in features:
        feature = process_feat(feature, 32)  # divide a video into 32 segments
        divided_features.append(feature)
    divided_features = np.array(divided_features, dtype=np.float32)
    return divided_features

def ClusterIndicesComp(indices_array, clusters):
    batch_array = []
    for i in range(clusters): 
        batch_array.append([j for j, x in enumerate(indices_array) if x == i])
    return np.array(batch_array, dtype='object')

class BatchSamplerNormal():
    def __init__(self):
        self.nor_ind = np.load('batch_nor_chunk_ind.npy', allow_pickle=True)
    def __iter__(self):
        # random.shuffle(self.nor_ind)
        return iter(self.nor_ind)
    
    def __len__(self):
        return (len(self.nor_ind))

class BatchSamplerAbnormal():
    def __init__(self):
        self.abn_ind = np.load('batch_abn_chunk_ind.npy', allow_pickle=True)
    def __iter__(self):
        # random.shuffle(self.abn_ind)
        return iter(self.abn_ind)
    
    def __len__(self):
        return (len(self.abn_ind))

def divide_n_a(batch_arr, rgb_list_file):
    n_arr = []
    a_arr = []
    lab = label_data_shanghai(rgb_list_file)
    for i in range(len(batch_arr)):
        temp_n = []
        temp_a = []
        for j in range(len(batch_arr[i])):
            if lab[batch_arr[i][j]] == 0:
                temp_n.append(batch_arr[i][j])
            else:
                temp_a.append(batch_arr[i][j])
        n_arr.append(temp_n)
        a_arr.append(temp_a)
    return np.array(n_arr, dtype="object"), np.array(a_arr, dtype="object")

def divide_chunks_great(arr_n, arr_a, chunk_size):
    div_arr_n = []
    div_arr_a = []
    for i in range(len(arr_n)):
        if len(arr_n[i])>chunk_size:
            div_arr_n.append(arr_n[i][:chunk_size])
            div_arr_n.append(arr_n[i][chunk_size:])
            if chunk_size-len(arr_a[i])>=0:
                div_arr_a.append(arr_a[i])
                div_arr_a.append(arr_a[i])
            elif chunk_size-len(arr_a[i])<0:
                div_arr_a.append(arr_a[i][:chunk_size])
                div_arr_a.append(arr_a[i][chunk_size:])
        elif len(arr_n[i])==chunk_size:
            div_arr_n.append(arr_n[i])
            if chunk_size-len(arr_a[i])>0:
                div_arr_a.append(arr_a[i])
            elif chunk_size-len(arr_a[i])<0:
                div_arr_a.append(arr_a[i][:chunk_size])
                div_arr_a.append(arr_a[i][chunk_size:])
                div_arr_n.append(arr_n[i])
            else:
                div_arr_a.append(arr_a[i])
    return div_arr_n, div_arr_a

def divide_chunks_less(arr_n, arr_a, chunk_size):
    for i in range(len(arr_n)):
        if chunk_size-len(arr_n[i])>=0:
            x = np.random.choice(arr_n[i], size=chunk_size-len(arr_n[i]), replace=True)
            if len(x)!=0:
                for e in x:
                    arr_n[i].append(e)

            if (len(arr_a[i]) == 0):
                while(len(arr_a[i])==0):
                    r = np.random.randint(len(arr_a))
                    arr_a[i] = arr_a[r]
            if chunk_size-len(arr_a[i])>0:
                y = np.random.choice(arr_a[i], size=chunk_size-len(arr_a[i]), replace=True)
                if(len(y)!=0):
                    for t in y:
                        arr_a[i].append(t)
    return arr_n, arr_a

if __name__ == '__main__':
    args = option.parser.parse_args()
    #batch_ind = np.load('batch_ind.npy', allow_pickle=True)
    rgb_list_file = list(open(args.rgb_list))
    print("len(rgb_list_file) ", len(rgb_list_file))
    clusters = len(rgb_list_file)//(2*args.batch_size) + 1
    cluster_ids_x, cluster_centers = cluster_shang(clusters, rgb_list_file)
    # #print(type(cluster_ids_x))
    batch_ind = ClusterIndicesComp(cluster_ids_x.detach().numpy(), clusters)
    np.save('batch_ind.npy', batch_ind, allow_pickle=True)
    print("num of clusters :", clusters)
    print(batch_ind)
    nor_arr, abn_arr = divide_n_a(batch_ind, rgb_list_file)
    print(nor_arr)
    print(abn_arr)
    np.save('batch_nor_ind.npy', nor_arr, allow_pickle=True)
    np.save('batch_abn_ind.npy', abn_arr, allow_pickle=True)
    nor_arr_chunked_g, abn_arr_chunked_g = divide_chunks_great(nor_arr, abn_arr, args.batch_size)
    print([len(nor_arr_chunked_g[i]) for i in range(len(nor_arr_chunked_g))])
    print([len(abn_arr_chunked_g[i]) for i in range(len(abn_arr_chunked_g))])
    print(nor_arr_chunked_g)
    print(abn_arr_chunked_g)
    # nor_arr_chunked_g, abn_arr_chunked_g = divide_chunks_great(nor_arr_chunked_g, abn_arr_chunked_g, args.batch_size)
    nor_arr_chunked_l, abn_arr_chunked_l = divide_chunks_less(nor_arr_chunked_g, abn_arr_chunked_g, args.batch_size)
    nor_arr_chunked_g, abn_arr_chunked_g = divide_chunks_great(nor_arr_chunked_l, abn_arr_chunked_l, args.batch_size)
    nor_arr_chunked_l, abn_arr_chunked_l = divide_chunks_less(nor_arr_chunked_g, abn_arr_chunked_g, args.batch_size)
    # nor_arr_chunked_g, abn_arr_chunked_g = divide_chunks_great(nor_arr_chunked_l, abn_arr_chunked_l, args.batch_size)
    # nor_arr_chunked_l, abn_arr_chunked_l = divide_chunks_less(nor_arr_chunked_g, abn_arr_chunked_g, args.batch_size)
    np.save('batch_nor_chunk_ind.npy', nor_arr_chunked_l, allow_pickle=True)
    np.save('batch_abn_chunk_ind.npy', abn_arr_chunked_l, allow_pickle=True)
    print([len(batch_ind[i]) for i in range(len(batch_ind))])
    print([len(nor_arr[i]) for i in range(len(batch_ind))])
    print([len(abn_arr[i]) for i in range(len(batch_ind))])
    print([len(nor_arr_chunked_g[i]) for i in range(len(nor_arr_chunked_g))])
    print([len(abn_arr_chunked_g[i]) for i in range(len(abn_arr_chunked_g))])
    print([len(nor_arr_chunked_l[i]) for i in range(len(nor_arr_chunked_l))])
    print([len(abn_arr_chunked_l[i]) for i in range(len(abn_arr_chunked_l))])
    print(nor_arr_chunked_g)
    print(abn_arr_chunked_g)
    print(nor_arr_chunked_l)
    print(abn_arr_chunked_l)
