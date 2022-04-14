import os
import pathlib
import cv2
import xmltodict
from VehicleDetection import VehicleDetection

def readFrameCount(sequence, camera):
    with open(f'./frame_counts/{sequence}.txt') as f:
        for line in f:
            if str(line[:3]) == camera:
                return line[4:]

def readDetectionsXML(path):
  #Generates detection dictionary where the frame number is the key and the values are the info of the corresponding detection/s
  
    with open(path,"r") as xml_obj:
        #coverting the xml data to Python dictionary
        gt = xmltodict.parse(xml_obj.read())
        #closing the file
        xml_obj.close()
    

    detections = {}
    for track in gt['annotations']['track']:
        if track['@label'] == 'car':
            for deteccion in track['box']:
                if deteccion['@frame'] not in detections:
                    detections[deteccion['@frame']] = []
                vh = VehicleDetection(int(deteccion['@frame']), int(track['@id']), 
                                                        float(deteccion['@xtl']), float(deteccion['@ytl']), 0, 0, 
                                                        1.0, float(deteccion['@xbr']), float(deteccion['@ybr']))
                if deteccion['attribute']['@name'] == 'parked' and deteccion['attribute']['#text'] == 'false':
                    vh.setParked(False)
                else:
                    vh.setParked(True)
                detections[deteccion['@frame']].append(vh)

    return detections


def getNotParkedCars(detections):
    notParked = {}
    for frame, objs in detections.items():
        obj_notParked = []
        for ob in objs:
            if not ob.parked:
                obj_notParked.append(ob)
        if len(obj_notParked) > 0:
            notParked[frame] = obj_notParked
    return notParked


def readDetections(path):
  #Generates detection dictionary where the frame number is the key and the values are the info of the corresponding detection/s
  
    with open(path) as f:
        lines = f.readlines()

    detections = {}
    for line in lines:
        data = line.split(',')
        if data[0] in detections:
            detections[data[0]].append(VehicleDetection(int(data[0]), int(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5]), float(data[6])))
        else:
            detections[data[0]] = [VehicleDetection(int(data[0]), int(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5]), float(data[6]))]

    return detections

def buildTrackCarsDataset():
    for seq in ['S01', 'S03', 'S04']:

    # get cam names from seq folder
        gt_path = f'../datasets/aic19-track1-mtmc-train/train/{seq}'
        dirlist = [ item for item in os.listdir(gt_path) if os.path.isdir(os.path.join(gt_path, item)) ]

        for cam in dirlist:
            gt_path = f'../datasets/aic19-track1-mtmc-train/train/{seq}/{cam}/gt/gt.txt'
            detections_list = readDetections(gt_path)

            prev = '0'
            for frame_num in detections_list:
                print(f'Processing Frame: {seq}_{cam}_{frame_num}')

                detections = detections_list[str(frame_num)]

                frame_path = f'../datasets/aic19-track1-mtmc-train/train/{seq}/{cam}/frames/vdo_{prev}.png'
                prev = frame_num

                img = cv2.imread(frame_path)

                if img is not None:
                    for detection in detections:
                        id = detection.ID
                        bbox = detection.getBBox()
                        size = detection.areaOfRec()

                        crop = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

                        pathlib.Path(f'aicity_cars_dataset/{id}/').mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(f'aicity_cars_dataset/{id}/{seq}_{cam}_{frame_num}.png', crop)