import os


'''
Escribe un archivo CSV para describir el dataset que se encuentra en routeToImages
Se supone que el dataset está organizado como

routeToImages
 - class1
    - image1
    - image2
 - class2
    - image1
    - image2

El CSV tendrá el formato
rutaImagen, etiqueta    
    
'''

routeToImages = "./train/"

classes = ["empty"]

CSVFile = "wwf.csv"


f = open(CSVFile, "w")

for idx, class_ in enumerate(classes):

    for root, dirs, files in os.walk(routeToImages + class_, topdown=True):
        for name in files:

            rutaIMG = os.path.join(root, name)

            f.write(rutaIMG + "," + str(idx) + "\n")

f.close()