import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import numpy as np
import cv2

# Define the helper function
def decode_segmap(image, nc=21): 
  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (255, 255, 255), (128, 128, 0), (255, 0, 0), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8) 
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]   
  rgb = np.stack([r, g, b], axis=2)
  return rgb
 

# load pth model
model = torch.load('model.pth')
# set model to inference mode
model.eval()
#print(model)

# prediction
img_path = 'robot2.jpg'
img = Image.open(img_path)
transform = T.Compose([T.Resize(520),
                   T.CenterCrop(480),
                   T.ToTensor(),
                   T.Normalize(        
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
                   ])

trans_img = transform(img).permute(1, 2, 0)
img = transform(img).unsqueeze(0)
out = model(img)['out']
print(img.shape)
print(out.shape)

om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
print (om.shape)
print (np.unique(om))
rgb = decode_segmap(om)
plt.subplot(121), plt.axis('off'), plt.imshow(trans_img)
plt.subplot(122), plt.axis('off'), plt.imshow(rgb)
plt.show()
