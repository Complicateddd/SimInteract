import argparse

from Detection import build_detector,Sort
import yaml
import cv2
from ProJ.util import plot_one_box,contrast_brightness_demo

import random
import numpy as np



if __name__ == "__main__":
        parser = argparse.ArgumentParser(description=' Pipeline Build Proj')
        parser.add_argument('--DetModelName', type=str, default='yolov5',
                            help='detector model yaml path')
        parser.add_argument('--DetModelCfg', type=str, default='./config/Detection/base/yolov5s.yaml',
                            help='detector model yaml path')
        parser.add_argument('--DetInferCfg', type=str, default='./config/Detection/inference/yolov5s.yaml',
                            help='detector Infer yaml path')

        opt = parser.parse_args()




        # check model name exist !!!
        with open('./config/default.yaml') as f:
                lib = yaml.safe_load(f)
        DetModelName = opt.DetModelName
        assert DetModelName in lib, "{} model is not implemented yet !!!".format(DetModelName)


        DetModelCfg = open(opt.DetModelCfg, 'r', encoding="utf-8")
        DetInferCfg = open(opt.DetInferCfg, 'r', encoding="utf-8")
        DetModelCfg = yaml.load(DetModelCfg, Loader=yaml.FullLoader)
        DetInferCfg = yaml.load(DetInferCfg, Loader=yaml.FullLoader)
        print(opt)
        print(DetModelCfg)
        print(DetInferCfg)


        # build detector
        detector = build_detector(DetModelName, DetModelCfg, DetInferCfg)
        ClassName = DetModelCfg['classname']

        # build sort tracker
        MoT_Tracker = Sort(10,8)



        # ColorsName = [[random.randint(0, 255) for _ in range(3)] for _ in range(50)]
        ColorsName = [[0,0,255] for _ in range(len(ClassName))]
        # ColorsName = np.random.randn((32,3))*255


        # Camera
        vid_cap = cv2.VideoCapture('/media/ubuntu/data/Simweights/sort3.mp4')
        # vid_cap = cv2.VideoCapture(0)

        vid_cap.set(6, cv2.VideoWriter.fourcc('M', 'J','P', 'G'))

        # vid_cap.set(cv2.CAP_PROP_FPS, 100)
        # print(vid_cap.get(cv2.CAP_PROP_FPS))
        vid_cap.set(3, 1080)
        vid_cap.set(4, 960)

        cv2.namedWindow("live", 0)
        # cv2.namedWindow("livedst", 0)

        cv2.moveWindow('live',200,150)
        # cv2.moveWindow('livedst',1000,150)


        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        size = (int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter('yolov5nano.mp4',fourcc,fps,size)


        # start live detection
        while True:
                ret, img0 = vid_cap.read()
                # dst = contrast_brightness_demo(img0, 0.3, 1) 
                # print(img0.shape)
                if ret:
                    '''+++++++++++++       detect pipeline         +++++++++++'''

                    det = detector.predict_box(img0)
                    
                    # print(tracker)
                    if det is not None and len(det):

                        det_array = np.array(det.cpu())[:,:5]
                        tracker = MoT_Tracker.update(det_array)

                        # for *xyxy, conf, cls in det:
                        #     x1, y1, x2, y2 = xyxy
                        #     # output_dict_.append((float(x1), float(y1), float(x2), float(y2)))
                        #     label = '%s %.2f' % (ClassName[int(cls)], conf)
                        #     plot_one_box(xyxy, img0, label=label, color=ColorsName[int(cls)], line_thickness=3)
                        # plot_one_box(xyxy, dst, label=label, color=ColorsName[int(cls)], line_thickness=2)

                        for *xyxy, cls in tracker:
                            x1, y1, x2, y2 = xyxy
                            # output_dict_.append((float(x1), float(y1), float(x2), float(y2)))
                            label = 'id {}'.format(int(cls))
                            plot_one_box(xyxy, img0, label=label, color=ColorsName[int(int(cls)%len(ClassName))], line_thickness=2)


                    cv2.resizeWindow("live", 1080,960)
                    # cv2.resizeWindow("livedst", 720,680)

                    cv2.imshow("live", img0)


                    # cv2.imshow("livedst", dst)
                    cv2.waitKey(15)

                    out.write(img0)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

        vid_cap.release()
        cv2.destroyAllWindows()
