import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
import cv2
import base64
import requests
def main():
  color = []
  i = "coffee"
  ldm = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-1", revision='fp16')
  if ldm.safety_checker is not None:
    ldm.safety_checker = lambda images, **kwargs: (images, False)
  image = ldm(i,height=256,width=256,num_inference_steps=27).images[0]
  img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
  color = []
  color.append(img.shape[0])
  color.append(img.shape[1])
  for y in range(img.shape[1]):
    for x in range(img.shape[0]):
      color.append(img[x, y, 2]*65536+img[x, y, 1]*256+img[x, y, 0])
  j = ','.join(map(str, color))
  n = 'p='+base64.b64encode(j.encode()).decode()
  r = requests.post("http://www.okaz0145.shop/projects/myapp/venv/lib/python3.6/site-packages/sample.py", data=n)
main()
