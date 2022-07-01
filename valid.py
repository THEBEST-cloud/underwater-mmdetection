from mmdet.apis import init_detector, inference_detector
import numpy as np


config_file = 'work_dirs/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_1x_coco/epoch_1.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
img = 'data/coco/water/guang_work/guang_forUser_A/forUser_A/val/image/c000005.jpg'
result = inference_detector(model, img)
# print(np.shape(result))
# print(result[1])
CLASSE = ['holothurian', 'echinus',
                 'scallop', 'starfish']


res_borderTable = [] #得到推理坐标、置信度

## for tables with borders
for j in range(len(result)):
    r = result[j]
    for i in range(len(r)):
        if r[i][-1]>.3:  
            res_borderTable.append([CLASSE[j],r[i][0].astype(int),r[i][1].astype(int),r[i][2].astype(int),r[i][3].astype(int),r[i][4].astype(float)])

print(res_borderTable)

model.show_result(img, result, out_file='demo/demo_result.jpg') # 保存推理图像