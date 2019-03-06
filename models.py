import cv2
from darkflow.net.build import TFNet
import argparse
import time
import numpy as np
import imutils
from utils.draw import set_ROI
from utils.optical_flow import Optical
from utils.helper import Utils, Helper


def init_paser():
    parser = argparse.ArgumentParser()
    # default : gia tri mac dinh cho tuy chon 
    # type : kieu gia tri se duoc convert
    # help: noi dung mo ta cho tham so
    parser.add_argument('--model', type=str,
                        default='cfg/yolo.cfg',
                        help='path to model config file')
    parser.add_argument('--video_folder', type=str,
                        default='images/bandem.mp4',
                        help='video path to dataset')
    parser.add_argument('--image', type=str,
                        default='images/000001.png',
                        help='image path to dataset')
    parser.add_argument('--load', type=str,
                        default="weights/yolov2.weights",
                        help='path to weights file')
    parser.add_argument('--config', type=str,
                        default="cfg",
                        help='path to config')
    parser.add_argument('--threshold', type=float,
                        default=0.5, help='object confidence threshold')
    parser.add_argument('--gpu', type=float,
                        default=0.0, help='Memory of GPU')
    parser.add_argument('--show_video', type=bool, default=True,
                        help='whether to use cuda if available')
    parser.add_argument('--test_image', type=bool, default=True,
                        help='defaults is test video')
    opt = parser.parse_args()
    return opt


class Model_yolov2:
    def __init__(self):
        self.opt = init_paser()

        # setup model
        self.option = {
            "model": self.opt.model,
            "load": self.opt.load,
            "threshold": self.opt.threshold,
            "gpu": self.opt.gpu,
            "config": self.opt.config
        }
        self.tfnet = TFNet(self.option)

        # set_ROI and create old_frame for tracking
        if not self.opt.test_image:
            self.capture = cv2.VideoCapture(self.opt.video_folder)
            ret, self.old_frame = self.capture.read()
            self.cor1, self.cor2, self.cor3, self.cor4, self.threshold_detection, self.threshold_delete = set_ROI(
                self.old_frame)
            self.old_gray = cv2.cvtColor(self.old_frame, cv2.COLOR_BGR2GRAY)
            self.tracker = Optical()
            # init some parameter for counting object
            self.number_frame = 0
            self.object1 = 0
            self.bike = 0
            self.capacity = 0
            self.number_object = np.array([0])

    def image(self, images=None):
        if images is not None:
            img = images
        else:
            img = cv2.imread(self.opt.image)
        results = self.tfnet.return_predict(img)
        print("RESULTS: %r" % results)
        #print(results)
        for result in results:
            tl, br, label, cf = Utils.get_coord(result)
            #print("CF: %r" % cf)
           #print("Utils.get_coord(result): %r" % Utils.get_coord(result))
            cv2.rectangle(img, tl, br, (255, 255, 0), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, label + " " + str(cf) , tl, font, 0.5,
                        (255, 255, 0), 2, cv2.LINE_AA)
        return images
        # if self.opt.show_video:
        #     cv2.imshow("img", img)
        #     cv2.waitKey(0)

    def counting_object(self):
        print(">>>>>>>>>>>>>>self.capture: %r" % self.capture)
        while (self.capture.isOpened()):
            stime = time.time()
            ret, frame = self.capture.read()
            self.old_gray = imutils.resize(self.old_gray, width=1280)
            frame = imutils.resize(frame, width=1280)
            results = []
            if ret:
                if self.number_frame % 1 == 0:
                    self.number_frame += 1
                    frame_gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
                    # get update the vector!
                    p1 = self.tracker.update(self.old_gray, frame_gray)
                    g_new = p1
                    g_old = self.tracker.p0

                    # draw the ROI
                    Utils.draw_ROI(frame, self.cor1, self.cor2,
                                   self.cor3, self.cor4)

                    # predict and return object is detected
                    results = self.tfnet.return_predict(frame)
                    for result in results:
                        helper = Helper(
                            result, self.cor1, self.cor2, self.cor3, self.cor4, self.threshold_detection)
                        p1, self.object1, self.number_object = helper.add_new_object(
                            frame, p1, self.object1, self.number_object)
                        p1, self.number_object = helper.del_out_of_ROI(
                            p1, self.threshold_delete, self.number_object)

                    # draw the point object is dectected
                    Utils.draw_object(frame, self.number_object, g_new, g_old)
                    print('FPS {:.1f}'.format(1 / (time.time() - stime)))
                    if self.opt.show_video:
                        cv2.imshow("frame", frame)

                    # Update old_fray and p0 for next flow1
                    self.old_gray = frame_gray.copy()
                    self.tracker.p0 = p1.reshape(-1, 1, 2)

                # press "q" to quit
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
            else:
                break
        self.capture.release()
        cv2.destroyAllWindows()
