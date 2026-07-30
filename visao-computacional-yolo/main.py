#imports
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import utils

#constantes
cfgPath = os.path.join(os.path.dirname(__file__), 'cfg', 'yolov3.cfg')
weightsPath = os.path.join(os.path.dirname(__file__), 'weights', 'yolov3.weights')
classesPath = os.path.join(os.path.dirname(__file__), 'data', 'coco.names')

print("-----------------------------------------------------------------------------")
print("Caso queira testar uma imagem própria basta salvá-la no formato .jpg e colocá-la na pasta 'images' que se encontra no mesmo nível deste arquivo.")

print("-----------------------------------------------------------------------------")
querScore = input("Você gostaria que o resultado final incluísse o score (o quão confiável é a detecção de cada objeto)? (S/N): ")

print("-----------------------------------------------------------------------------")
nomeImg = input("Digite o nome do arquivo da imagem que deseja testar (exemplo: cat ou cat.jpg): ")

print("-----------------------------------------------------------------------------")
nmsUser = float(input("Digite o valor, entre 0 e 1, que deseja usar como threshold do nms (exemplo: 0.7): "))

print("-----------------------------------------------------------------------------")
iouUser = float(input("Digite o valor, entre 0 e 1, que deseja usar como threshold do iou (exemplo: 0.5): "))

if '.jpg' in nomeImg:
    imagePath = os.path.join(os.path.dirname(__file__), 'images', nomeImg)
else:
    imagePath = os.path.join(os.path.dirname(__file__), 'images', nomeImg + '.jpg')


#lendo as classes

with open(classesPath, 'r') as f:
    classes = [j[:-1] for j in f.readlines() if len(j) > 2]
    f.close()

#carregando o modelo
rede = cv2.dnn.readNetFromDarknet(cfgPath, weightsPath)

#carregando a imagem
img = cv2.imread(imagePath)

#arrumando a imagem
imgAjustada = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), True)

#detecção
rede.setInput(imgAjustada) #passando a imagem para a rede
saida = utils.resultado(rede) #pegando os resultados

#salvando bounding boxes, ids e scores
caixas = []
idClasses = []
scores = []

height, width, _ = img.shape

for s in saida:

    caixa = s[:4]
    xc, yc, w, h = caixa #coordenadas e tamanho da caixa

    confiancaCaixa = s[4] #pc (probabilidade de ter um objeto qualquer)

    caixa = [int(xc * width), int(yc * height), int(w * width), int(h * height)] #voltando a caixa para o tamanho absoluto
    caixas.append(caixa)

    id = np.argmax(s[5:]) #pegando o id da classe mais provável
    idClasses.append(id)

    score = np.amax(s[5:])*confiancaCaixa #multiplicando a probabilidade de classe mais alta pela confianca da caixa (o score)
    scores.append(score)

#aplicar nms e iou

caixas, idClasses, scores = utils.nms(caixas, idClasses, scores, nmsUser, iouUser)

dicionarioCores = dict()

#mostrar resultado
for i, caixa in enumerate(caixas):
    
    classeAtual = idClasses[i]

    if classeAtual in dicionarioCores:
        cor = dicionarioCores[classeAtual]
    else:
        cor = np.random.randint(0, 255, size=(3,1))
        dicionarioCores[classeAtual] = cor

    xc, yc, w, h = caixa

    img = cv2.rectangle(img,
                        (int(xc - (w / 2)), int(yc - (h / 2))),
                        (int(xc + (w / 2)), int(yc + (h / 2))),
                        (int(cor[0]), int(cor[1]), int(cor[2])),
                        2)
    
    if(querScore.upper() == 'S'):
        texto = classes[idClasses[i]] + ":" + str(round(scores[i], 2))
    else:
        texto = classes[idClasses[i]]

    (font_width, font_height), baseline = cv2.getTextSize(texto, cv2.FONT_HERSHEY_DUPLEX, 1, 1)
    
    img = cv2.rectangle(img,
                        (max(0,int(xc - (w / 2)-20)), max(0,int(yc - (h / 2) - font_height))),
                        (min(width,int(xc - (w / 2) + font_width)), min(height,int(yc - (h / 2)))),
                        (int(cor[0]), int(cor[1]), int(cor[2])),
                        -1)

    cv2.putText(img,
            texto,
            (max(0,int(xc - (w / 2)-20)), max(0,int(yc - (h / 2)))),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (0, 0, 0),
            1)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()