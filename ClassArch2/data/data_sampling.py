import random
import os
import shutil


BASE_DIR = "../data/modelnet40-normal_numpy/"
train_file = BASE_DIR + "train_all.txt"
labeled_file = BASE_DIR + "train_files.txt"
unlabeled_file = BASE_DIR + "unlabeled_files.txt"
labels_file = BASE_DIR + "shape_names.txt"


def initDir(path, dir_list, som_dir_list):
    if os.path.exists(path):
        shutil.rmtree(path)

    os.mkdir(path)
    for dir in dir_list+som_dir_list:
        os.mkdir(os.path.join(path, dir))

    for som_dir in som_dir_list:
        for dir in dir_list:
            os.mkdir(os.path.join(path, som_dir, dir))


def writeFile(path, data):
    with open(path, "w") as f:
        for line in data:
            f.write(line + "\n")
            

if __name__=='__main__':
    # Remove previous files and init sub directories
    labels = []
    with open(labels_file, "r") as labels_f:
        lines = labels_f.readlines()
        for line in lines:
            labels.append(line[:-1])

    som_dir_list = []
    for i in range(3, 12):
        som_dir_list.append("%dx%d_som_nodes" %(i, i))

    initDir(os.path.join(BASE_DIR, "train"), labels, som_dir_list)
    initDir(os.path.join(BASE_DIR, "unlabeled"), labels, som_dir_list)


    # Separate total train data to labeled/unlabeled data
    train_data = []
    with open(train_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            train_data.append(line[:-1])

    train_cnt_list = [0 for _ in range(40)]
    labeled_cnt_list = []
    for i, label in enumerate(labels):
        for data in train_data:
            if label in data:
                train_cnt_list[i] += 1
    
    for train_cnt in train_cnt_list:
        labeled_cnt_list.append(round(train_cnt / 5 * 3))

    print("Original train files: %d" %sum(train_cnt_list))
    print("Sampling.. labeled:unlabeled = 3:2")
    print("train files: %d" %sum(labeled_cnt_list))
    print("unlabeled files: %d" %(sum(train_cnt_list) - sum(labeled_cnt_list)))

    labeled_data = []
    unlabeled_data = []
    for i, label in enumerate(labels):
            for data in train_data:
                if label in data:
                    if labeled_cnt_list[i] != 0:
                        labeled_data.append(data) 
                        labeled_cnt_list[i] -= 1
                    else:
                        unlabeled_data.append(data)


    # unlabeled_data = random.sample(train_data, unlabeled_cnt)
    # labeled_data = list(set(train_data) - set(unlabeled_data))

    writeFile(labeled_file, labeled_data)
    writeFile(unlabeled_file, unlabeled_data)
        
    
    # Move separated labeled/unlabeled data to each directory 
    for data in labeled_data:
        label = data[:-5]
        file_name = data+".npy"
        
        origin_path = os.path.join(BASE_DIR, "original", label, file_name)
        copy_path = os.path.join(BASE_DIR, "train", label, file_name)
        shutil.move(origin_path, copy_path)

        for som_dir in som_dir_list:
            origin_path = os.path.join(BASE_DIR, "original", som_dir, label, file_name)
            copy_path = os.path.join(BASE_DIR, "train", som_dir, label, file_name)
            shutil.move(origin_path, copy_path)

    for data in unlabeled_data:
        label = data[:-5]
        file_name = data+".npy"

        origin_path = os.path.join(BASE_DIR, "original", label, file_name)
        copy_path = os.path.join(BASE_DIR, "unlabeled", label, file_name)
        shutil.move(origin_path, copy_path)

        for som_dir in som_dir_list:
            origin_path = os.path.join(BASE_DIR, "original", som_dir, label, file_name)
            copy_path = os.path.join(BASE_DIR, "unlabeled", som_dir, label, file_name)
            shutil.move(origin_path, copy_path)



