import numpy as np
import cv2

def resultado(rede):
    camadas = rede.getLayerNames()
    camadasSaida = [camadas[i - 1] for i in rede.getUnconnectedOutLayers()]
    saidas = rede.forward(camadasSaida) #processamento
    saidas = [s for saida in saidas for s in saida if s[4]>0.05]
    return saidas

def nms(boundingBoxes, idClasses, scores, nms_thresh = 0.8, iou_thresh = 0.5):
    boundingBoxes = np.asarray(boundingBoxes)
    idClasses = np.asarray(idClasses)
    scores = np.asarray(scores)

    indicesRemover = [] 

    for i in range(0, len(scores)):
        if scores[i] < nms_thresh:
            indicesRemover.append(i)

    boundingBoxes = np.delete(boundingBoxes, indicesRemover, axis = 0)
    idClasses = np.delete(idClasses, indicesRemover)
    scores = np.delete(scores, indicesRemover)

    if len(boundingBoxes) == 0:
        return [], [], []

    indices = np.arange(len(scores))

    removerIou = []

    for i, box in enumerate(boundingBoxes):
        classeAtual = idClasses[i]

        iTemp = indices[indices!=i]
             
        x1 = box[0] - (box[2] / 2)
        y1 = box[1] - (box[3] / 2)
        x2 = box[0] + (box[2] / 2)
        y2 = box[1] + (box[3] / 2)
        areaAtual = (x2-x1)*(y2-y1)
        
        x1Inter = np.maximum(x1, boundingBoxes[iTemp, 0] - (boundingBoxes[iTemp, 2] / 2))
        y1Inter = np.maximum(y1, boundingBoxes[iTemp, 1] - (boundingBoxes[iTemp, 3] / 2))
        x2Inter = np.minimum(x2, boundingBoxes[iTemp, 0] + (boundingBoxes[iTemp, 2] / 2))
        y2Inter = np.minimum(y2, boundingBoxes[iTemp, 1] + (boundingBoxes[iTemp, 3] / 2))
    
        wInter = np.maximum(0, x2Inter - x1Inter)
        hInter = np.maximum(0, y2Inter - y1Inter)
        areasInter = wInter*hInter

        x1Outros = boundingBoxes[iTemp, 0] - (boundingBoxes[iTemp, 2] / 2)
        y1Outros = boundingBoxes[iTemp, 1] - (boundingBoxes[iTemp, 3] / 2)
        x2Outros = boundingBoxes[iTemp, 0] + (boundingBoxes[iTemp, 2] / 2)
        y2Outros = boundingBoxes[iTemp, 1] + (boundingBoxes[iTemp, 3] / 2)

        areaOutros = (x2Outros-x1Outros)*(y2Outros-y1Outros)
        areasUniao = areaOutros+areaAtual-areasInter

        overlap = areasInter / areasUniao
        
        for j in range(len(overlap)):
            if classeAtual == idClasses[iTemp[j]]:
                if overlap[j] > iou_thresh:
                    if scores[i] > scores[iTemp[j]]:
                        if iTemp[j] not in removerIou:
                            removerIou.append(iTemp[j])
    indices = np.delete(indices, removerIou)

    return boundingBoxes[indices], idClasses[indices], scores[indices]