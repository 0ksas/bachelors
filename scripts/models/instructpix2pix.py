import requests
import torch
from io import BytesIO

from diffusers import StableDiffusionInstructPix2PixPipeline
import transformers
from PIL import Image
import glob
import os

model_id_or_path = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16).to("cuda")
generator = torch.Generator("cuda").manual_seed(0)

resolution = "512"
testImagePath = "Dataset/Combined_V2/testA/*.png"
testImageResultPath = "results/InstructPix2Pix_" + resolution
prompt = "turn into Vice City style"
num_inference_steps = 60
image_guidance_scale = 1.5
guidance_scale = 10

for filename in glob.glob(testImagePath + '/*.jpg'):
  init_image = Image.open(filename).convert("RGB")
  init_image = init_image.resize((512, 512))

  edited_image = pipe(
    prompt,
    image=init_image,
    num_inference_steps=num_inference_steps,
    image_guidance_scale=image_guidance_scale,
    guidance_scale=guidance_scale,
    generator=generator,
  ).images[0]
  edited_image.save(testImageResultPath + "/" + os.path.basename(filename))