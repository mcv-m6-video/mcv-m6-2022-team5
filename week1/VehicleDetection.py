import cv2


#Class to represent vehicle detections
class VehicleDetection:
    def __init__(self, frame, ID, left, top, width, height, conf):
        self.frame = frame
        self.ID   = ID
        self.xtl  = left
        self.ytl  = top
        self.xbr  = left + width
        self.ybr  = top + height
        self.conf = conf
        self.w = width
        self.h = height

    def drawRectangleOnImage(self, img, color=(0, 255, 0)):
        cv2.rectangle(img, (self.xtl, self.ytl), (self.xbr, self.ybr), color)

        return img

    def areaOfRec(self):
        return self.w * self.h

    def areaOfIntersection(self, detec2):
        # Format of the bboxes is [tly, tlx, bry, brx, ...], where tl and br
        # indicate top-left and bottom-right corners of the bbox respectively.

        # determine the coordinates of the intersection rectangle
        xA = max(self.xtl, detec2.xtl)
        yA = max(self.ytl, detec2.ytl)
        xB = min(self.xbr, detec2.xbr)
        yB = min(self.ybr, detec2.ybr)
        
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)

        return interArea


    def areaOfUnion(self, detec2):
        intersectionArea = self.areaOfIntersection(detec2)

        return self.areaOfRec() + detec2.areaOfRec() - intersectionArea


    def IoU(self, detec2):
        return self.areaOfIntersection(detec2) / self.areaOfUnion(detec2)

    def getBBox(self):
        return [self.xtl, self.ytl, self.xbr, self.ybr]

    def __str__(self) -> str:
        return f'Frame {self.frame}, TL [{self.xtl},{self.ytl}], BR [{self.xbr},{self.ybr}], Confidence {self.conf}'