from datasets import load_dataset
from PIL import ImageFilter, Image
import tqdm

ds = load_dataset('lambdalabs/pokemon-blip-captions')['train']
i = 0

for image in tqdm.tqdm(ds):
    # Convert the image to grayscale
    gray_image = image.convert('L')

    # Apply the Sobel filter for edge detection
    image_canny = gray_image.filter(ImageFilter.CONTOUR)
    
    image_canny = image_canny.resize((512,512))

    image_canny.save('1.dataset\\dataset\\input\\{:03d}.jpg'.format(i))
    image.save('1.dataset\\dataset\\output\\{:03d}.jpg'.format(i))
    
    i += 1