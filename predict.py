#-----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#-----------------------------------------------------------------------#
import time

import cv2
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from PIL import Image

from yolo import YOLO


def False_Boxes_Remove(results):
    # 此处有问题，下标不存在问题
    if (results[0][0] is None)or(results[1][0] is None):
        return results
    else:
        P1boxes = results[0][0][:, :4]
        P2boxes = results[1][0][:, :4]
    # print(P1boxes)
    # print(type(P1boxes))
    # print(len(P1boxes))
    if len(P1boxes) and len(P2boxes):

        p1removeboxlist=list()
        p2removeboxlist=list()
        if len(P1boxes) <= len(P2boxes):
            for box1 in P1boxes:
                for box2 in P2boxes:
                    # print(box1)
                    # print(type(box1))
                    # print(p1removeboxlist)
                    # print(type(p1removeboxlist))
                    if compute_iou(box1, box2) > 0.8:
                        # if box1.tolist() in p1removeboxlist:
                        #     pass
                        # else:
                        #     p1removeboxlist.append(box1)
                        # if box2.tolist() in p2removeboxlist:
                        #     pass
                        # else:
                        #     p2removeboxlist.append(box2)
                        # if  box2 in p2removeboxlist:
                        #     pass
                        # else:
                        #     p2removeboxlist.append(box2)
                        # print(type(p1removeboxlist))
                        # print(type(box1))
                        # print('box1 = ', box1.tolist())
                        # print('p1removeboxlist = ', p1removeboxlist)
                        cnt1 = 0
                        for p1list in p1removeboxlist:
                            if set(box1.tolist()) <= set(p1list):
                                cnt1 += 1
                        if cnt1 == 0:
                            p1removeboxlist.append(box1.tolist())
                        cnt2 = 0
                        for p2list in p2removeboxlist:
                            if set(box2.tolist()) <= set(p2list):
                                cnt2 += 1
                        if cnt2 == 0:
                            p2removeboxlist.append(box2.tolist())
                        # if  set(box1.tolist()) <= set(p1removeboxlist):
                        #     pass
                        # else:
                        #     p1removeboxlist.append(box1.tolist())
                        # if  box2 in p2removeboxlist:
                        #     pass
                        # else:
                        #     p2removeboxlist.append(box2.tolist())

        else:
            for box2 in P2boxes:
                for box1 in P1boxes:
                    if compute_iou(box1, box2) > 0.8:
                        cnt1 = 0
                        for p1list in p1removeboxlist:
                            if set(box1.tolist()) <= set(p1list):
                                cnt1 += 1
                        if cnt1 == 0:
                            p1removeboxlist.append(box1.tolist())
                        cnt2 = 0
                        for p2list in p2removeboxlist:
                            if set(box2.tolist()) <= set(p2list):
                                cnt2 += 1
                        if cnt2 == 0:
                            p2removeboxlist.append(box2.tolist())
    # print(p1removeboxlist)
    # print(p2removeboxlist)
    # result0removelist = list()
    # result1removelist = list()
    r0list = results[0][0].tolist()
    r1list = results[1][0].tolist()
    for i in p1removeboxlist:
        #P1boxes.remove(i)
        # 直接在results里删除
        for j in results[0][0]:
            # print(i)
            # print(j)
            # print(j[:4])
            # print(type(j[:4]))
            if all(i==j[:4]):

                r0list.remove(j.tolist())
                # result0removelist.append(np.where(results[0][0]==j))
                # results[0][0] = np.delete(results[0][0],np.where(results[0][0]==j))
                # print(results)
                # results[0][0].remove(j)
    for i in p2removeboxlist:


        #P2boxes.remove(i)
        #直接在results里删除
        for j in results[1][0]:
            if all(i==j[:4]):
                r1list.remove(j.tolist())
                # result1removelist.append(np.where(results[1][0] == j))
                # results[1][0] = np.delete(results[1][0],np.where(results[1][0]==j))
                # results[1][0].remove(j)
                # print(results)
    results[0][0] = np.array(r0list)
    results[1][0] = np.array(r1list)
    return results
def compute_iou(rec1, rec2):
        """
        computing IoU
        :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
        :param rec2: (y0, x0, y1, x1)
        :return: scala value of IoU
        """
        # computing area of each rectangles
        S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

        # computing the sum_area
        sum_area = S_rec1 + S_rec2

        # find the each edge of intersect rectangle
        left_line = max(rec1[1], rec2[1])
        right_line = min(rec1[3], rec2[3])
        top_line = max(rec1[0], rec2[0])
        bottom_line = min(rec1[2], rec2[2])

        # judge if there is an intersect
        if left_line >= right_line or top_line >= bottom_line:
            return 0
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)
            return (intersect / (sum_area - intersect)) * 1.0
if __name__ == "__main__":
    yolo = YOLO()
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    #----------------------------------------------------------------------------------------------------------#
    #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
    #   想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
    #   想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps用于保存的视频的fps
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #-------------------------------------------------------------------------#
    #   test_interval用于指定测量fps的时候，图片检测的次数
    #   理论上test_interval越大，fps越准确。
    #-------------------------------------------------------------------------#
    test_interval   = 100
    #-------------------------------------------------------------------------#
    #   dir_origin_path指定了用于检测的图片的文件夹路径
    #   dir_save_path指定了检测完图片的保存路径
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "/kaggle/working/img_out/"

    if mode == "predict":
        '''
        1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''
        ImgList = list()
        img0 = input('Input image filename1:')
        img1 = input('Input image filename2:')
        ImgList.append(Image.open(img0))
        ImgList.append(Image.open(img1))
        pool = ThreadPool()
        start = time.time()
        results = pool.map(yolo.detect_image, ImgList)
        # 所有结果
        # print(type(results))
        # print('\n')
        print(results)
        # # # print('\n')
        # # # print(results[0])
        # # # print('\n')
        # print(results[0])
        # if results[0][0] is None:
        #     print('真')
        # # print(len(results[0]))
        # print(results[0][0])
        # print(len(results[0][0]))
        # print(results[0][0][0])
        # # 标签类别
        # print(np.array(results[0][0][:, 6], dtype = 'int32'))
        # # 精确度
        # print(results[0][0][:4])
        # print(results[0][0][:,:4])
        # print(results[0][0][:, 5])
        #print(results)
        # 框坐标
        # p1boxes = results[0][0][:, :4]
        # print(p1boxes)
        # p2boxes = results[1][0][:, :4]
        # print(len(p1boxes))
        # print(len(p2boxes))
        # IOU阈值判断
        # print(results)
        # print(results[0])
        # results = np.delete(results,np.where(results==results[0]))
        # print(results)
        # print(type(results[0][0]))
        results = False_Boxes_Remove(results)
        end = time.time()
        runtime = end - start
        print(runtime)
        # print(results)
        # print(results[0][0])
        # print(type(results[0][0]))
        r_image = yolo.draw_PIC(results[0],ImgList[0])
        r_image.show()
        r_image1 = yolo.draw_PIC(results[1],ImgList[1])
        r_image1.show()
        print('\n')
        pool.close()
        pool.join()
















































    #     while True:
    #         img = input('Input image filename:')
    #         try:
    #             image = Image.open(img)
    #         except:
    #             print('Open Error! Try again!')
    #             continue
    #         else:
    #             r_image = yolo.detect_image(image)
    #             r_image.show()
    # elif mode == "video":
    #     capture = cv2.VideoCapture(video_path)
    #     if video_save_path!="":
    #         fourcc  = cv2.VideoWriter_fourcc(*'XVID')
    #         size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #         out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
    #
    #     fps = 0.0
    #     while(True):
    #         t1 = time.time()
    #         # 读取某一帧
    #         ref,frame=capture.read()
    #         # 格式转变，BGRtoRGB
    #         frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #         # 转变成Image
    #         frame = Image.fromarray(np.uint8(frame))
    #         # 进行检测
    #         frame = np.array(yolo.detect_image(frame))
    #         # RGBtoBGR满足opencv显示格式
    #         frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    #
    #         fps  = ( fps + (1./(time.time()-t1)) ) / 2
    #         print("fps= %.2f"%(fps))
    #         frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #
    #         cv2.imshow("video",frame)
    #         c= cv2.waitKey(1) & 0xff
    #         if video_save_path!="":
    #             out.write(frame)
    #
    #         if c==27:
    #             capture.release()
    #             break
    #     capture.release()
    #     out.release()
    #     cv2.destroyAllWindows()
    #
    # elif mode == "fps":
    #     img = Image.open('img/street.jpg')
    #     tact_time = yolo.get_FPS(img, test_interval)
    #     print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
    #
    # elif mode == "dir_predict":
    #     import os
    #     from tqdm import tqdm
    #
    #     img_names = os.listdir(dir_origin_path)
    #     for img_name in tqdm(img_names):
    #         if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
    #             image_path  = os.path.join(dir_origin_path, img_name)
    #             image       = Image.open(image_path)
    #             r_image     = yolo.detect_image(image)
    #             if not os.path.exists(dir_save_path):
    #                 os.makedirs(dir_save_path)
    #             r_image.save(os.path.join(dir_save_path, img_name))
    #
    # else:
    #     raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
