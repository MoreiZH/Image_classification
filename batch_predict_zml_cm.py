import os
import json
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from models.my_ResNet50 import resnet50
from datasets.dataloder import LoadDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


def get_most_confused_per_class_map(cm_per_df):
    cm_per_true_dict = {}  # confusion matrix with true label
    cm_per_predict_dict = {}  # confusion matrix with predicted label
    for i in range(len(cm_per_df)):
        d_ind_l = np.argsort(-cm_per_df[i, :]).tolist()  # z to a order
        temp_sum = cm_per_df[i, :].sum()
        if temp_sum != 0:
            cm_per_true_dict['真实标签' + str(i)] = [(index, np.round(cm_per_df[i, index] / temp_sum, 4))
                                                 for index in d_ind_l]
        else:
            cm_per_true_dict['真实标签' + str(i)] = 0
        d_ind_l = np.argsort(-cm_per_df[:, i]).tolist()
        temp_sum = cm_per_df[:, i].sum()
        if temp_sum != 0:
            cm_per_predict_dict['预测标签' + str(i)] = [(index, np.round(cm_per_df[index, i] / temp_sum, 4))
                                                    for index in d_ind_l]
        else:
            cm_per_predict_dict['预测标签' + str(i)] = 0
    return cm_per_true_dict, cm_per_predict_dict


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # data_loader
    in_f = './datasets/test.csv'
    img_path_df = pd.read_csv(in_f)
    img_path_list = img_path_df['img_path'].values.tolist()
    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    testset = LoadDataset(in_f, img_transform=data_transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                             num_workers=nw, drop_last=False)
    print("using {} images for test.".format(len(testset)))

    # read class_indict
    json_path = './output/torchID_wiki.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = resnet50(num_classes=len(class_indict)).to(device)

    # load model weights
    weights_path = "./output/resNet50.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    with torch.no_grad():
        true_label_list = []
        predicted_label_list = []
        temp_cc_list = []  # cc:confidence coefficient
        cc_per_true_dict = dict([('真实标签' + str(label_digit), [])
                                 for label_digit in range(len(class_indict))])
        for pos, (data, label) in enumerate(test_loader):
            data = data.to(device)
            label = label.to(device)
            score = model(data)
            probs, predicted = torch.max(score.data, 1)
            s_matrix = F.softmax(score.data, dim=1)

            sm_size = s_matrix.size()
            for i in range(sm_size[0]):
                temp_cc_list.append(s_matrix[i][predicted[i]].cpu().numpy())
            true_label_list.extend(list(label.cpu().numpy()))
            predicted_label_list.extend(list(predicted.cpu().numpy()))

        for index in range(len(temp_cc_list)):
            true_label = true_label_list[index]
            predicted_label = predicted_label_list[index]
            cc = float(temp_cc_list[index])
            cc_per_true_dict['真实标签' + str(true_label)].append((int(predicted_label), '{:.4f}'.format(cc)))
        # 生成混淆矩阵
        testset_c_matrix = confusion_matrix(true_label_list, predicted_label_list)
        cm_per_true_dict, cm_per_predict_dict = get_most_confused_per_class_map(testset_c_matrix)
        # 输出混淆矩阵
        out_cm_1 = './output/测试集中真实标签最易被混淆成的其余标签排序.csv'
        cm_per_true_df = pd.DataFrame(cm_per_true_dict, index=['真实标签来源, 概率'] * len(class_indict))
        cm_per_true_df.to_csv(out_cm_1)

        out_cm_1 = './output/测试集中其余标签最易被混淆成预测标签的排序.csv'
        cm_per_predict_df = pd.DataFrame(cm_per_predict_dict, index=['真实标签来源, 概率'] * len(class_indict))
        cm_per_predict_df.to_csv(out_cm_1)

        result_df = pd.DataFrame(columns=['图片名称', '预测类别', '置信度'])
        for idx, (pro, cla) in enumerate(zip(probs, predicted)):
            result_df.loc[idx] = [os.path.split(img_path_list[idx])[-1],
                                  class_indict[str(cla.cpu().numpy())],
                                  round(float(pro.cpu().numpy()), 4)]
            # print("image: {}  class: {}  prob: {:.3}".format(img_path_list[idx],
            #                                                  class_indict[str(cla.numpy())],
            #                                                  pro.numpy()))
        out_f = './output/' + os.path.split(in_f)[-1].split('.')[0] \
                + os.path.split(weights_path)[-1].split('.')[0] + '预测结果.csv'
        result_df.to_csv(out_f, index=False, encoding="UTF-8-sig")
        print('预测结果已经保存在{}！'.format(out_f))


if __name__ == '__main__':
    main()
