import cv2
import imutils

class CoordinateStore:
    def __init__(self, img):
        self.points = []
        self.img = img

    def select_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(self.img, (x, y), 3, (255, 0, 0), -1)
            self.points.append((x, y))

    def setROI(self):
        cor1 = self.points[0]
        cor2 = self.points[1]
        cor3 = (self.points[2][0], cor2[1])
        cor4 = (self.points[3][0], cor1[1])
        threshold_detection = self.points[1][1]
        threshold_delete = self.points[0][1]
        return cor1, cor2, cor3, cor4, threshold_detection, threshold_delete


def set_ROI(img):
    img = imutils.resize(img, width=1280)
    coord = CoordinateStore(img)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', coord.select_point)
    while(1):
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('q') or len(coord.points) > 3:
            break
    cv2.destroyAllWindows()
    return coord.setROI()
