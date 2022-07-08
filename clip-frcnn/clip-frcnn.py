import numpy as np
import torch
from pkg_resources import packaging
import clip
import tqdm
from PIL import Image
import os, json
from tqdm import tqdm
import skimage
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
from torchvision.datasets import CIFAR100

#from transformers import CLIPProcessor, CLIPModel

device = 'cpu'
model, preprocess = clip.load('RN50x4', device = device, jit = False)
#model = CLIPModel.from_pretrained('RN50x4')
#processor = CLIPProcessor.from_pretrained('RN50x4')
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

test_images = os.path.join(os.getcwd(), 'test_images')
file = open('caption_dict.json')
caption_dict = json.load(file)
caption_dict = dict(list(caption_dict.items()))
caption_dict_0 = dict(list(caption_dict.items())[:len(caption_dict)//2])
caption_dict_1 = dict(list(caption_dict.items())[len(caption_dict)//2:])

capt_higher_than_foil = 0
f_h_t_c = 0
p_c_h_t_f = 0
p_f_h_t_c = 0

for item_id, info in tqdm(caption_dict.items()):
    #print(item_id, info)
    # print(item_id)
    # print(info['passive'], info['foils'])
    if info['passive'] != 'NOT AVAILABLE' and info['passive_foil'] != 'NOT AVAILABLE':
        print(info['passive'], info['passive_foil'])
        image_path = os.path.join(test_images, info['image_file'])
        image = Image.open(image_path)
        test_sentences = [info['passive'], info['passive_foil']]

        image = preprocess(image).unsqueeze(0).to(device)
        text = clip.tokenize(test_sentences).to(device)
        
        #inputs = preprocess(text=test_sentences, images=image, return_tensors="pt", padding=True)

        # outputs = model(**inputs)
        # logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        # probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

        # image_input = torch.tensor(preprocess(image))
        # text_tokens = clip.tokenize(test_sentences)

        with torch.no_grad():
            image_features = model.encode_image(image)#image_input).float()
            text_features = model.encode_text(text)#.float()

        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        print("Label probs:", probs)

        #cifar100 = CIFAR100(os.path.expanduser("~/.cache"), transform=preprocess, download=True)

        # image_ = processor(image).unsqueeze(0).to(device)

        #logits_per_image, logits_per_text = model(**inputs)

        # print(logits_per_image)
        # print(logits_per_text)

        info['lxmert'] = dict()
        info['lxmert']['PCAF'] = {0:None, 1:None}
        info['lxmert']['PCAF'][0] = probs[0][0].item()
        info['lxmert']['PCAF'][1] = probs[0][1].item()

        if probs[0][0].item() > probs[0][1].item():
            p_c_h_t_f += 1
        else:
            p_f_h_t_c += 1


    if info['caption'] != 'NOT AVAILABLE' and info['foils'] != 'NOT AVAILABLE':
        image_path = os.path.join(test_images, info['image_file'])
        image = Image.open(image_path)
        test_sentences = [info['caption'], info['foils']]
        
        #inputs = preprocess(text=test_sentences, images=image, return_tensors="pt", padding=True)

        image = preprocess(image).unsqueeze(0).to(device)
        text = clip.tokenize(test_sentences).to(device)

        # outputs = model(**inputs)
        # logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        # probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

        # print(outputs)

        with torch.no_grad():
            image_features = model.encode_image(image)#image_input).float()
            text_features = model.encode_text(text)#.float()

        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        if info.get('lxmert', False) == False:
            info['lxmert'] = dict()
        info['lxmert']['foil'] = {0:None, 1:None}
        info['lxmert']['foil'][0] = probs[0][0].item()
        info['lxmert']['foil'][1] = probs[0][1].item()

        if probs[0][0].item() > probs[0][1].item():
            capt_higher_than_foil += 1
        else:
            f_h_t_c += 1
    
    

print('captions higher than foils:', capt_higher_than_foil)
print('foils higher than captions:', f_h_t_c)
print('passive captions higher than foils:', p_c_h_t_f)
print('passive foils higher than captions:', p_f_h_t_c)

with open(f'PCAF_clip_original_rn.json', 'w') as outfile:
     json.dump(caption_dict, outfile, indent=4)