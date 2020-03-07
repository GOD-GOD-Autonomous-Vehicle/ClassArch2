from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.utils.data as data
import numpy as np
import os
import h5py
import subprocess
import shlex
import math
import faiss 

BASE_DIR = './'


def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip()[5:] for line in f]


def _load_data_file(name):
    f = h5py.File(name)
    data = f["data"][:]
    label = f["label"][:]
    return data, label


class ModelNet40(data.Dataset):
    def __init__(self, num_points, transforms=None, split='train', download=False):
        super().__init__()

        self.transforms = transforms

        self.folder = "modelnet40_ply_hdf5_2048"
        self.data_dir = os.path.join(BASE_DIR, self.folder)
        self.url = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"

        if download and not os.path.exists(self.data_dir):
            zipfile = os.path.join(BASE_DIR, os.path.basename(self.url))
            # subprocess.check_call(
            #     shlex.split("wget {} -o {}".format(self.url, zipfile))
            # )

            # subprocess.check_call(
            #     shlex.split("unzip {} -d {}".format(zipfile, BASE_DIR))
            # )

            # subprocess.check_call(shlex.split("rm {}".format(zipfile)))
            os.system('wget https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip')
            print("DATA DOWNLOADED")
            os.system('unzip modelnet40_ply_hdf5_2048.zip')
            print("DATA UNZIPPED")

        self.split = split
        if self.split == 'train':
            self.files = _get_data_files(os.path.join(self.data_dir, "train_files.txt"))
        elif self.split == 'test':
            self.files = _get_data_files(os.path.join(self.data_dir, "test_files.txt"))

        point_list, label_list = [], []
        for f in self.files:
            points, labels = _load_data_file(os.path.join(BASE_DIR, f))
            point_list.append(points)
            label_list.append(labels)

        self.points = np.concatenate(point_list, 0)
        self.labels = np.concatenate(label_list, 0)
        self.set_num_points(num_points)

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.num_points)
        np.random.shuffle(pt_idxs)

        current_points = self.points[idx, pt_idxs].copy()
        label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)

        if self.transforms is not None:
            current_points = self.transforms(current_points)

        return current_points, label

    def __len__(self):
        return self.points.shape[0]

    def set_num_points(self, pts):
        self.num_points = min(self.points.shape[1], pts)

    def randomize(self):
        pass



class ModelNet40_SONet(data.Dataset):
    def __init__(self, root, mode, opt, transforms=None):
        super().__init__()
        self.root = root
        self.opt = opt
        self.mode = mode
        self.transforms = transforms

        self.dataset = []
        rows = round(math.sqrt(opt.node_num))
        cols = rows

        f = open(os.path.join(root, 'shape_names.txt'))
        shape_list = [str.rstrip() for str in f.readlines()]
        f.close()

        if 'train' == mode:
            f = open(os.path.join(root, 'train_files.txt'), 'r')
            lines = [str.rstrip() for str in f.readlines()]
            f.close()
        elif 'test' == mode:
            f = open(os.path.join(root, 'test_files.txt'), 'r')
            lines = [str.rstrip() for str in f.readlines()]
            f.close()
        elif 'unlabeled' == mode:
            f = open(os.path.join(root, 'unlabeled_files.txt'), 'r')
            lines = [str.rstrip() for str in f.readlines()]
            f.close()
        elif 'train_stu' == mode:
            f = open(os.path.join(root, 'train_stu_files.txt'), 'r')
            lines = [str.rstrip() for str in f.readlines()]
            f.close()
        else:
            raise Exception('Network mode error.')

        for i, name in enumerate(lines):
            # locate the folder name
            folder = name[0:-5]
            file_name = name

            # get the label
            label = shape_list.index(folder)

            # som node locations
            som_nodes_folder = '%dx%d_som_nodes' % (rows, cols)

            item = (os.path.join(root, mode, folder, file_name + '.npy'),
                    label,
                    os.path.join(root, mode, som_nodes_folder, folder, file_name + '.npy'))
            self.dataset.append(item)

        # kNN search on SOM nodes
        self.knn_builder = KNNBuilder(self.opt.som_k)

        # farthest point sample
        self.fathest_sampler = FarthestSampler()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.opt.dataset == 'modelnet':
            pc_np_file, class_id, som_node_np_file = self.dataset[index]

            data = np.load(pc_np_file)
            data = data[np.random.choice(data.shape[0], self.opt.input_pc_num, replace=False), :]

            pc_np = data[:, 0:3]  # Nx3
            surface_normal_np = data[:, 3:6]  # Nx3
            som_node_np = np.load(som_node_np_file)  # node_numx3
            
        elif self.opt.dataset == 'shrec':
            npz_file, class_id = self.dataset[index]
            data = np.load(npz_file)

            pc_np = data['pc']
            surface_normal_np = data['sn']
            som_node_np = data['som_node']

            # random choice
            choice_idx = np.random.choice(pc_np.shape[0], self.opt.input_pc_num, replace=False)
            pc_np = pc_np[choice_idx, :]
            surface_normal_np = surface_normal_np[choice_idx, :]
        else:
            raise Exception('Dataset incorrect.')

        # if self.transforms is not None: # Need to be fixed
        #     pc_np = self.transforms(pc_np)

        # convert to tensor
        pc = torch.from_numpy(pc_np.transpose().astype(np.float32))  # 3xN

        # surface normal
        surface_normal = torch.from_numpy(surface_normal_np.transpose().astype(np.float32))  # 3xN

        # som
        som_node = torch.from_numpy(som_node_np.transpose().astype(np.float32))  # 3xnode_num

        # kNN search: som -> som
        if self.opt.som_k >= 2:
            D, I = self.knn_builder.self_build_search(som_node_np)
            som_knn_I = torch.from_numpy(I.astype(np.int64))  # node_num x som_k
        else:
            som_knn_I = torch.from_numpy(np.arange(start=0, stop=self.opt.node_num, dtype=np.int64).reshape((self.opt.node_num, 1)))  # node_num x 1

        # print(som_node_np)
        # print(D)
        # print(I)
        # assert False
        
        return pc, surface_normal, class_id, som_node, som_knn_I



class UnlabeledModelNet40(data.Dataset):
    def __init__(self, num_points, root, transforms=None, split='unlabeled'):
        super().__init__()
        self.transforms = transforms
        root = os.path.abspath(root)
        self.folder = "modelnet40_ply_hdf5_2048"
        self.data_dir = os.path.join(root, self.folder)

        self.split, self.num_points = split, num_points
        if self.split == 'unlabeled':
            self.files =  _get_data_files( \
                os.path.join(self.data_dir, 'unlabeled_files.txt'))

        point_list = []
        for f in self.files:
            points, _ = _load_data_file(os.path.join(root, self.files[-1]))
            point_list.append(points)

        self.points = np.concatenate(point_list, 0)

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])
        
        current_points = self.points[idx, pt_idxs].copy()
        
        if self.transforms is not None:
            current_points = self.transforms(current_points)
        
        current_points = torch.transpose(current_points, 1, 0)
        
        return current_points

    def __len__(self):
        return self.points.shape[0]



class KNNBuilder:
    def __init__(self, k):
        self.k = k
        self.dimension = 3

    def build_nn_index(self, database):
        '''
        :param database: numpy array of Nx3
        :return: Faiss index, in CPU
        '''
        index = faiss.IndexFlatL2(self.dimension)  # dimension is 3
        index.add(database)
        return index

    def search_nn(self, index, query, k):
        '''
        :param index: Faiss index
        :param query: numpy array of Nx3
        :return: D: numpy array of Nxk
                 I: numpy array of Nxk
        '''
        D, I = index.search(query, k)
        return D, I

    def self_build_search(self, x):
        '''

        :param x: numpy array of Nxd
        :return: D: numpy array of Nxk
                 I: numpy array of Nxk
        '''
        x = np.ascontiguousarray(x, dtype=np.float32)
        index = self.build_nn_index(x)
        D, I = self.search_nn(index, x, self.k)
        return D, I


class FarthestSampler:
    def __init__(self):
        pass

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def sample(self, pts, k):
        farthest_pts = np.zeros((k, 3))
        farthest_pts[0] = pts[np.random.randint(len(pts))]
        distances = self.calc_distances(farthest_pts[0], pts)
        for i in range(1, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(distances, self.calc_distances(farthest_pts[i], pts))
        return farthest_pts

