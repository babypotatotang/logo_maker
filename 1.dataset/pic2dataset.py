import pandas as pd 
from PIL import Image
import io 

def img2byte(image):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes = image_bytes.getvalue()
    
    return image_bytes

input_folder = 'D:\\1. Project\\4.Tobigs_Conference\\1.dataset\\raw_data\\input'
output_folder = 'D:\\1. Project\\4.Tobigs_Conference\\1.dataset\\raw_data\\output'

df = pd.read_csv('D:\\1. Project\\4.Tobigs_Conference\\1.dataset\\pokemon_blip2_t5_caption_coco_flant5xl.csv')
captions = list(df['0'])

df = pd.DataFrame()

for i in range(833):
    image = '{:03d}.jpg'.format(i)
    
    input = input_folder + "\\" + image
    output = output_folder + "\\" + image 

    input_img = img2byte(Image.open(input).convert('RGB').resize((512, 512)))
    output_img = img2byte(Image.open(output).convert('RGB').resize((512, 512)))
    caption = captions[i]
    
    data = {
        'input_image': input_img,
        'edit_prompt': caption,
        'edited_image': output_img
    }
    
    df = df._append(data, ignore_index = True)

df.to_parquet('D:\\1. Project\\4.Tobigs_Conference\\1.dataset\\20230530_pokemon_blip2_t5_caption_coco_flant5xl.parquet')
    