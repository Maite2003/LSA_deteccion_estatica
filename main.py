#from ultralytics import YOLOv11
#import opencsv

from images import process_images

# model = YOLOv11.from_pretrained("Ultralytics/YOLO11")
# source = 'http://images.cocodataset.org/val2017/000000039769.jpg'

images = []
for i in range(1, 6):
  img_route = f'dataset/001_00{i}.png'
  images.append(img_route)
  
results = process_images(images)

print(results[0])


#model.predict(source=source, save=True)