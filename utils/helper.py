from math import sqrt
import cv2
import numpy as np


class Utils:

    def get_coord(result):
        tl = (result['topleft']['x'], result['topleft']['y'])
        br = (result['bottomright']['x'], result['bottomright']['y'])
        label = result['label']
        cf = result['confidence']
        print("confidence: %r" % cf)
        return tl, br, label, cf

    def draw_ROI(frame, cor1, cor2, cor3, cor4):
        cv2.line(frame, cor1, cor2, (255, 255, 0), 2)
        cv2.line(frame, cor1, cor4, (255, 255, 0), 2)
        cv2.line(frame, cor2, cor3, (255, 255, 0), 2)
        cv2.line(frame, cor3, cor4, (255, 255, 0), 2)

    def draw_object(frame, number_object, g_new, g_old):
        for i, (new, old) in enumerate(zip(g_new, g_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            if i < len(number_object):
                cv2.putText(frame, str(
                    number_object[i]), (c, d), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), thickness=3)


class Helper:

    def __init__(self, result, cor1, cor2, cor3, cor4, threshold):
        self.tl, self.br, self.label = Utils.get_coord(result)
        self.point_flow = self.point_center()
        self.cor1 = cor1
        self.cor2 = cor2
        self.cor3 = cor3
        self.cor4 = cor4
        self.threshold = threshold

    def box_large(self):
        return self.br[0] - self.tl[0]

    def point_center(self):
        return ((self.tl[0] + self.br[0]) / 2, (self.tl[1] + self.br[1]) / 2 + 0.25 * (self.br[1] - self.tl[1]))

    def distance(self, p1, p2):
        return sqrt((p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]))

    def sign(self, cor1, cor2, x, y):
        return (cor2[1] - cor1[1]) * (x - cor1[0]) - (cor2[0] - cor1[0]) * (y - cor1[1])

    def check_in_ROI(self):
        return (self.sign(self.cor1, self.cor4, self.point_flow[0], self.point_flow[1]) < 0 and
                self.sign(self.cor3, self.cor4, self.point_flow[0], self.point_flow[1]) > 0 and
                self.sign(self.cor1, self.cor2, self.point_flow[0], self.point_flow[1]) > 0 and
                self.sign(self.cor2, self.cor3, self.point_flow[0], self.point_flow[1]) > 0 and
                self.point_flow[1] <= self.threshold)

    def add_new_object(self, frame, p1, object1, number_object):
        check = self.check_in_ROI()
        if check and self.label != "" and self.box_large() < 250:
            cv2.rectangle(frame, self.tl, self.br, (0, 255, 0), 1)
            chk = False
            for i in p1:
                # distance between point_flow and all point in p1
                print(self.distance(self.point_flow, (i[0][0], i[0][1])))
                if self.distance(self.point_flow, (i[0][0], i[0][1])) < 50:
                    chk = True
            if not chk:
                p2 = np.array([[[self.point_flow[0], self.point_flow[1]]]],
                              dtype=np.float32)
                p1 = np.append(p1, p2, axis=0)
                object1 += 1
                number_object = np.append(number_object, object1)
        return p1, object1, number_object

    def del_out_of_ROI(self, p1, threshold, number_object):
        size_of_p1 = len(p1)
        index_cur = 0
        while index_cur < size_of_p1:
            if int(p1[index_cur][0][1]) <= threshold:
                p1 = np.delete(p1, index_cur, axis=0)
                size_of_p1 -= 1
                number_object = np.delete(number_object, index_cur)
            else:
                index_cur += 1
        if len(p1) < 1:
            p1 = np.array([[[0, 0]]], dtype=np.float32)
        if len(number_object) < 1:
            number_object = np.array([0])
        return p1, number_object
