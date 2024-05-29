import os
import random
import xml.etree.ElementTree as ET
import numpy as np
from utils.utils import get_classes
import glob
"""
0代表整个标签处理过程，包括获得VOCdevkit/VOC2007/ImageSets里面的txt以及训练用的2007_train.txt、2007_val.txt
1代表获得VOCdevkit/VOC2007/ImageSets里面的txt
2代表获得训练用的2007_train.txt、2007_val.txt

python voc_yolo_txt_Exdark_test.py
"""
annotation_mode = 0
# 仅在annotation_mode为0和2的时候有效
classes_path = '/export/yuanzhian/lujiajia/paper_code_1/low_light_object_detection/yolov5-pytorch-main/model_data/classes_Exdark.txt'
# 仅在annotation_mode为0和1的时候有效
# trainval_percent =0.9                  # 用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1
# train_percent = 0.9                      # 用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1

VOCdevkit_path = '/export/yuanzhian/lujiajia/paper_code_1/object_detection_dataset/Exdark/ExDARk_YOLO3_training'
VOCdevkit_sets = [('Exdark', 'train'), ('Exdark', 'val'),('Exdark', 'test')]

classes, _ = get_classes(classes_path)
photo_nums = np.zeros(len(VOCdevkit_sets))
nums = np.zeros(len(classes))

#xml转换为txt格式
def convert_annotation(image_set, image_id, list_file):
    # in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/SegmentationClass/%s.xml' % (year, image_id)), encoding='utf-8')
    txtfile=glob.glob(os.path.join('/export/yuanzhian/lujiajia/paper_code_1/object_detection_dataset/Exdark/ExDARk_YOLO3_training/labels/%s'%image_set, image_id+'*'))[0]
    #/home/lujiajia/paper_code_1/object_detection_dataset/Exdark/ExDARk_YOLO3_training/labels/train/2015_01557.txt

    with open(txtfile, 'r') as ftxt:
        lines = ftxt.readlines()
    for line in lines:
        # print(line)#3 0.2484375 0.028125 0.2921875 0.95
        line_l=line.strip('\n').split(' ')#['3', '0.2484375', '0.028125', '0.2921875', '0.95']
        '''
                l - 顶部x坐标
                t - 顶部y坐标
                w - bounding box的宽
                h - bounding box的高
                Exdark [xmin,ymin,width,height] 需要转换为 [xmin,ymin,xmax,ymax]格式
        '''
        cls_id=line_l[0]
        line_l[1]=line_l[1]
        line_l[2]=line_l[2]
        line_l[3]=str(int(line_l[1])+int(line_l[3]))
        line_l[4]=str(int(line_l[2])+int(line_l[4]))
        
        list_file.write(" " + ",".join([str(a) for a in line_l[1:5]]) + ',' + str(cls_id))
        # list_file.write(line)
    # nums[cls_id] = nums[cls_id] + 1


if __name__ == "__main__":
    random.seed(0)
    train_annotation_path   = '/export/yuanzhian/lujiajia/paper_code_1/object_detection_dataset/Exdark/ExDARk_YOLO3_training/labels/train'
    val_annotation_path     = '/export/yuanzhian/lujiajia/paper_code_1/object_detection_dataset/Exdark/ExDARk_YOLO3_training/labels/val'
    train_img_path='/export/yuanzhian/lujiajia/paper_code_1/object_detection_dataset/Exdark/ExDARk_YOLO3_training/images/train'
    val_img_path='/export/yuanzhian/lujiajia/paper_code_1/object_detection_dataset/Exdark/ExDARk_YOLO3_training/images/val'
    test_img_path='/export/yuanzhian/lujiajia/paper_code_1/object_detection_dataset/Exdark/ExDARk_YOLO3_training/images/test'
    if " " in os.path.abspath(VOCdevkit_path):
        raise ValueError("数据集存放的文件夹路径与图片名称中不可以存在空格，否则会影响正常的模型训练，请注意修改。")
    if annotation_mode == 0 or annotation_mode == 1:#保存训练集、验证集、测试集txt
        print("Generate txt in ImageSets.")
        image_name_train=os.listdir(train_img_path)
        image_name_test=os.listdir(test_img_path)
        image_name_val=os.listdir(val_img_path)
        # xmlfilepath     = os.path.join(VOCdevkit_path, 'VOC2007/Annotations')
        saveBasePath    = '/export/yuanzhian/lujiajia/paper_code_1/object_detection_dataset/Exdark/ExDARk_YOLO3_training'

        num_train = len(image_name_train) 
        num_test=len(image_name_test) 
        num_val=len(image_name_val) 
        
        #保存训练集、验证集、测试集的图片id
        # ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
        ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
        ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
        fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
        for j in range(num_train):
            name=image_name_train[j].split('.')[0]+'\n'
            ftrain.write(name)
        for j in range(num_val):
            name=image_name_val[j].split('.')[0]+'\n'
            fval.write(name)
          
        for j in range(num_test):
            name=image_name_test[j].split('.')[0]+'\n'
            ftest.write(name)
        
        # ftrainval.close()  
        ftrain.close()  
        fval.close()  
        ftest.close()
        print('训练集多少张：',num_train)# 3000
        print('验证集多少张：',num_val)# 1800
        print('测试集多少张：',num_test)#2563
        print("Generate txt in ImageSets done.")
    if annotation_mode == 0 or annotation_mode == 2:
        print("Generate Exdark_train.txt and Exdark_val.txt  for train.")
        #2007_train.txt and 2007_val.txt存放：图片的路径、四个位置、类别ID
        type_index = 0
        for year, image_set in VOCdevkit_sets:
            image_ids=[]
            # image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Segmentation/%s.txt' % (year, image_set)),
            #                  encoding='utf-8').read().strip().split()
            if image_set=='train':
                image_name=os.listdir(train_img_path)
                for i in image_name:
                    image_ids.append(i.split('.')[0])
            elif image_set=='val':
                image_name=os.listdir(val_img_path)
                for i in image_name:
                    image_ids.append(i.split('.')[0])
            else:
                image_name=os.listdir(test_img_path)
                for i in image_name:
                    image_ids.append(i.split('.')[0])
            
            list_file = open('%s_%s.txt' % (year, image_set), 'w', encoding='utf-8')
            for image_id in image_ids:
                img_file_path = glob.glob(os.path.join('/export/yuanzhian/lujiajia/paper_code_1/object_detection_dataset/Exdark/ExDARk_YOLO3_training/images/%s'%image_set, image_id+'*'))[0]
                #/home/lujiajia/paper_code_1/object_detection_dataset/Exdark/ExDARk_YOLO3_training/images/train/2015_01557.jpg
                list_file.write(img_file_path)#图片的路径
                convert_annotation(image_set, image_id, list_file)
                list_file.write('\n')
              
            photo_nums[type_index] = len(image_ids)
            type_index += 1
            list_file.close()
        print("Generate Exdark_train.txt and Exdark_val.txt for train done.")


        # def printTable(List1, List2):
        #     for i in range(len(List1[0])):
        #         print("|", end=' ')
        #         for j in range(len(List1)):
        #             print(List1[j][i].rjust(int(List2[j])), end=' ')
        #             print("|", end=' ')
        #         print()


        # str_nums = [str(int(x)) for x in nums]
        # tableData = [
        #     classes, str_nums
        # ]
        # colWidths = [0] * len(tableData)
        # len1 = 0
        # for i in range(len(tableData)):
        #     for j in range(len(tableData[i])):
        #         if len(tableData[i][j]) > colWidths[i]:
        #             colWidths[i] = len(tableData[i][j])
        # printTable(tableData, colWidths)

        if photo_nums[0] <= 500:
            print("训练集数量小于500，属于较小的数据量，请注意设置较大的训练世代（Epoch）以满足足够的梯度下降次数（Step）。")

        if np.sum(nums) == 0:
            print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")

