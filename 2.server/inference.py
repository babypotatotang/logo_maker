from PIL import Image
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

model_id = "/home/s20235025/tobigs-pokemon/build" # <- replace this 
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
generator = torch.Generator("cuda").manual_seed(0)

pipe.safety_checker = lambda images, clip_input: (images, False)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

image = Image.open('/home/s20235025/tobigs-pokemon/test.jpg')
image = image.convert('RGB')

prompt = "a character wearing red and blue pants"
num_inference_steps = 30
image_guidance_scale = 1.5
guidance_scale = 30

edited_image = pipe(prompt, 
   image=image, 
   num_inference_steps=num_inference_steps, 
   image_guidance_scale=image_guidance_scale, 
   guidance_scale=guidance_scale,
   generator=generator,
).images[0]
edited_image.save("tobigs-pokemon/test_edited2.png")