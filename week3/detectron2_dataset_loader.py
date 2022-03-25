import xmltodict


#TODO: Not finisehd, only a esqueleton
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
                
                if deteccion['attribute']['@name'] == 'parked' and deteccion['attribute']['#text'] == 'false':
                    pass
                else:
                    pass
                detections[deteccion['@frame']].append(None)

    return detections