import os

from mmdet.apis import init_detector, inference_detector
import numpy as np
import torch
from tqdm import tqdm

if __name__ == "__main__":
    # 构建model
    device = torch.device("cuda:6")
    config_file = 'work_dirs/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_1x_coco/epoch_12.pth'
    model = init_detector(config_file, checkpoint_file, device=device)  # or device='cuda:0'
    CLASSE = ['holothurian', 'echinus', 'scallop', 'starfish']
    
    test_img_path = "data/coco/water/guang_work/guang_forUser_A/forUser_A/test-A-image"
    for img_name in tqdm(os.listdir(test_img_path)):
        # print(img_name)
        img_path = os.path.join(test_img_path,img_name)
        txt_name = img_name.split(".jpg")[0]+".txt"
        # print(txt_name)
        txt_path = os.path.join("sub_optics",txt_name)
        
        result = inference_detector(model, img_path)
        output = open(txt_path,'w',encoding='gbk')

        res_borderTable = [] #得到推理坐标、置信度
        ## for tables with borders
        for j in range(len(result)):
            r = result[j]
            for i in range(len(r)):
                if r[i][-1]>.3:  
                    res_borderTable.append([CLASSE[j],r[i][0].astype(int),r[i][1].astype(int),r[i][2].astype(int),r[i][3].astype(int),r[i][4].astype(float)])
        
        for row in res_borderTable:
            rowtxt = '{},{},{},{},{},{:.2f}'.format(row[0],row[1],row[2],row[3],row[4],row[5])
            output.write(rowtxt)
            output.write('\n')
        output.close()
    
    print("********************************************************************")
    print("                                 Done!                              ")
    print("********************************************************************")
