import requests
import torch
from io import BytesIO

from diffusers import CycleDiffusionPipeline, DDIMScheduler
import transformers
from PIL import Image
import glob

import os

model_id_or_path = "CompVis/stable-diffusion-v1-4"
scheduler = DDIMScheduler.from_pretrained(model_id_or_path, subfolder="scheduler")
pipe = CycleDiffusionPipeline.from_pretrained(model_id_or_path, scheduler=scheduler).to("cuda")

resolution = "512"
testImagePath = "Dataset/Combined_V2/testA/*.png"
testImageResultPath = "results/CycleDiffusion_" + resolution
source_prompt = "Real life car traffic"
prompt = "Vice City car traffic"

print("starting")
for filename in glob.glob(testImagePath):
  print(filename)
  init_image = Image.open(filename).convert("RGB")
  init_image = init_image.resize((512, 512))

  image = pipe(
      prompt=prompt,
      source_prompt=source_prompt,
      image=init_image,
      num_inference_steps=100,
      eta=0.1,
      strength=0.8,
      guidance_scale=3,
      source_guidance_scale=2,
  ).images[0]

  image.save(testImageResultPath + "/" + os.path.basename(filename))