import xmltodict
from VehicleDetection import VehicleDetection

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