from PIL import Image
import requests
import os
import json
import tqdm
from tqdm import tqdm
import clip
import torch

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

test_images = os.path.join(os.getcwd(), 'test_images')
file = open('caption_dict.json')
caption_dict = json.load(file)
caption_dict = dict(list(caption_dict.items()))
caption_dict_0 = dict(list(caption_dict.items())[:len(caption_dict)//2])
caption_dict_1 = dict(list(caption_dict.items())[len(caption_dict)//2:])


###
# device = 'cpu'
# inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image # this is the image-text similarity score
# logits_per_text = outputs.
# probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
###
# print(probs)
# print(probs[0])
# print(probs[0][0].item())

capt_higher_than_foil = 0
f_h_t_c = 0
p_c_h_t_f = 0
p_f_h_t_c = 0

for item_id, info in tqdm(caption_dict.items()):
    print(item_id, info)
    if info['caption'] != 'NOT AVAILABLE' and info['foils'] != 'NOT AVAILABLE':
        
        image_path = os.path.join(test_images, info['image_file'])
        image = Image.open(image_path)
        test_sentences = [info['caption'], info['foils']]
        
        inputs = processor(text=test_sentences, images=image, return_tensors="pt", padding=True)

        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities


        # image_ = processor(image).unsqueeze(0).to(device)

        #logits_per_image, logits_per_text = model(**inputs)

        # print(logits_per_text)

        info['lxmert'] = dict()
        info['lxmert']['PCAF'] = {0:None, 1:None}
        info['lxmert']['PCAF'][0] = probs[0][0].item()
        info['lxmert']['PCAF'][1] = probs[0][1].item()

        if probs[0][0].item() > probs[0][1].item():
            capt_higher_than_foil += 1
        else:
            f_h_t_c += 1

    if info['passive'] != 'NOT AVAILABLE' and info['passive_foil'] != 'NOT AVAILABLE':
        image_path = os.path.join(test_images, info['image_file'])
        image = Image.open(image_path)
        test_sentences = [info['foils'], info['passive_foil']]
        
        inputs = processor(text=test_sentences, images=image, return_tensors="pt", padding=True)

        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

        #print(outputs)

        if info.get('lxmert', False) == False:
            info['lxmert'] = dict()
        info['lxmert']['foil'] = {0:None, 1:None}
        info['lxmert']['foil'][0] = probs[0][0].item()
        info['lxmert']['foil'][1] = probs[0][1].item()

        if probs[0][0].item() > probs[0][1].item():
            p_c_h_t_f += 1
        else:
            p_f_h_t_c += 1

print('captions higher than foils:', capt_higher_than_foil)
print('foils higher than captions:', f_h_t_c)
print('passive captions higher than foils:', p_c_h_t_f)
print('passive foils higher than captions:', p_f_h_t_c)

# with open(f'PCAF_clip_pc_vit.json', 'w') as outfile:
#     json.dump(caption_dict, outfile, indent=4)