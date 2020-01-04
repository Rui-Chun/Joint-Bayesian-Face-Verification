import os
from PIL import Image
import numpy as np
from common import *
from joint_bayesian import *


def excute_train(train_path="../lfw-origin", result_fold="../result/"):
    train_data = []
    train_labels = []
    dir_list = os.listdir(train_path)
    for idx in range(len(dir_list)):
        dir_name = dir_list[idx]
        if dir_name[0] == '.':
            continue
        path = os.path.join(train_path, dir_name)
        for img_name in os.listdir(path):
            img = Image.open(os.path.join(path, img_name)).convert('L')
            img = img.resize((150, 150))
            train_data.append(np.array(img).flatten()/255)  # Do we need flatten???
            train_labels.append(idx)
    train_data = np.array(train_data, dtype=float)

    # data predeal
    data = data_pre(train_data)

    # pca training.
    # pca = PCA_Train(data, result_fold)
    with open(result_fold + '/pca_model.pkl', 'rb') as f:
        pca = pickle.load(f)
    data_pca = pca.transform(data)

    data_to_pkl(data_pca, result_fold+"pca_lfw_train.pkl")
    data_pca = pickle.load("pca_lfw_train.pkl")
    JointBayesian_Train(data_pca, train_labels, result_fold)


def excute_test_new(pair_path = "../LFW", result_fold="../result/"):
    with open(result_fold+"A_eval.pkl", "rb") as f:
        A = pickle.load(f)
    with open(result_fold+"G_eval.pkl", "rb") as f:
        G = pickle.load(f)

    print("Start Test, A, G loaded")

    test_data = []
    test_label = []
    dir_list0 = os.listdir(pair_path)
    for idx in range(len(dir_list0)):
        dir_name = dir_list0[idx]
        if dir_name[0] == '.':
            continue
        path0 = os.path.join(pair_path, dir_name)
        dir_list1 = os.listdir(path0)
        for idy in range(len(dir_list1)):
            if dir_list1[idy][0] == '.':
                continue
            path1 = os.path.join(path0, dir_list1[idy])
            for img_name in os.listdir(path1):
                img = Image.open(os.path.join(path1, img_name)).convert('L')
                img = img.resize((150, 150))
                test_data.append(np.array(img).flatten()/255)  # Do we need flatten???
            if dir_name == 'match pairs':
                test_label.append(1)
            else:
                test_label.append(0)
    test_data = np.array(test_data, dtype=float)
    test_label = np.array(test_label)
    assert 2*test_label.shape[0] == test_data.shape[0], 'Test Data Shape Wrong'
    data = data_pre(test_data)
    print("Test data loaded")

    # pca_model = joblib.load(result_fold+"pca_model.m")
    with open(result_fold+'/pca_model.pkl', 'rb') as f:
        pca_model = pickle.load(f)
    data = pca_model.transform(data)
    print("PCA processed")

    dist_all = get_pair_ratios(A, G, data)

    assert len(dist_all)==len(test_label), 'wrong num of label'

    dist_copy = dist_all.copy()
    np.random.shuffle(dist_copy)
    threshold_ratio = np.median(dist_copy[0:1000])

    # Get threshold！！
    # threshold_ratio = np.median(dist_all)
    # print('threshold = {} saved'.format(threshold_ratio))
    # thre_file = open(os.path.join(result_fold, 'thre.pkl'), 'wb')
    # pickle.dump(threshold_ratio, thre_file)
    # thre_file.close()
    thre_file = open(os.path.join(result_fold, 'thre.pkl'), 'rb')
    threshold_ratio = pickle.load(thre_file)
    thre_file.close()
    print('threshold = {} loaded'.format(threshold_ratio))

    TP=0; TN=0; FP=0; FN=0
    for item in range(len(dist_all)):
        if dist_all[item] > threshold_ratio and test_label[item]:
            TP+=1
        elif dist_all[item] < threshold_ratio and test_label[item]==0:
            TN +=1
        elif dist_all[item] > threshold_ratio and test_label[item]==0:
            FP+=1
        elif dist_all[item] < threshold_ratio and test_label[item]:
            FN+=1

    recall = TP/(TP + FN)
    # accu = (TP + TN) / (TP + FP + FN + TN)
    prec = TP / (TP + FP)
    f1_score = 2 * (recall * prec) / (recall + prec)

    print("total test num= {}, TP = {}; TN = {}; FP = {}; FN = {} ".format((TP + TN + FP + FN), TP, TN, FP, FN))
    print('Recall={}, Prec={}, F1 score={}'.format(recall, prec, f1_score))

def excute_eval(eval_path = "../LFW_eval", result_file = "../results.txt", result_fold="../result/"):
    with open(result_fold + "A_eval.pkl", "rb") as f:
        A = pickle.load(f)
    with open(result_fold + "G_eval.pkl", "rb") as f:
        G = pickle.load(f)

    eval_data = []

    dir_list = os.listdir(eval_path)
    dir_list.sort()
    for idy in range(len(dir_list)):
        if dir_list[idy][0]=='.':
            continue

        path1 = os.path.join(eval_path, dir_list[idy])
        for img_name in os.listdir(path1):
            img = Image.open(os.path.join(path1, img_name)).convert('L')
            img = img.resize((150, 150))
            eval_data.append(np.array(img).flatten() / 255)  # Do we need flatten???

    test_data = np.array(eval_data, dtype=float)

    data = data_pre(test_data)

    # pca_model = joblib.load(result_fold + "pca_model.m")
    with open(result_fold+'/pca_model.pkl', 'rb') as f:
        pca_model = pickle.load(f)
    data = pca_model.transform(data)

    print("Eval Data Loaded")

    dist_all = get_pair_ratios(A, G, data)

    thre_file = open(os.path.join(result_fold, 'thre.pkl'), 'rb')
    threshold_ratio = pickle.load(thre_file)
    thre_file.close()
    print('threshold = {} loaded'.format(threshold_ratio))

    results_out = np.where(dist_all > threshold_ratio, 1, 0)
    out_file = open(result_file, "w")
    for item in range(len(results_out)):
        out_file.write("{}\n".format(results_out[item]))
    out_file.close()


if __name__ == "__main__":
    # excute_train()
    excute_test_new()
    # excute_eval()
    pass
