from images import process_images


# hago yo la funcion pero deberia existir alguna que ya haga esto
def transformar_a_string(n):
  respuesta = ''
  if (n < 10):
    respuesta = f'00{n}'
  else:
    respuesta = f'0{n}'
  return respuesta

# images = []
# for i in range(1, 11):
#   senia = transformar_a_string(i)
#   for j in range(1,9):  
#     toma = transformar_a_string(j)
#     img_route = f'dataset/{senia}_{toma}.png'
#     images.append(img_route)
  
# results = process_images(images)



from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
train_results = model.train(
    data="LSA80.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("path/to/image.jpg")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model
