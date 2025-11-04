#from ultralytics import YOLOv11
#import opencsv

from images import process_images

# model = YOLOv11.from_pretrained("Ultralytics/YOLO11")
# source = 'http://images.cocodataset.org/val2017/000000039769.jpg'

# hago yo la funcion pero deberia existir alguna que ya haga esto
def transformar_a_string(n):
  respuesta = ''
  if (n < 10):
    respuesta = f'00{n}'
  else:
    respuesta = f'0{n}'
  return respuesta

images = []
for i in range(1, 11):
  senia = transformar_a_string(i)
  for j in range(1,9):  
    toma = transformar_a_string(j)
    img_route = f'dataset/{senia}_{toma}.png'
    images.append(img_route)
  
results = process_images(images)

#model.predict(source=source, save=True)