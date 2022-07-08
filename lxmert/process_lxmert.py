from IPython.display import clear_output, Image, display
import torch
from processing_image import Preprocess
from modeling_frcnn import GeneralizedRCNN
from transformers import LxmertForQuestionAnswering, LxmertTokenizer, LxmertForPreTraining
import os, json
import random
from tqdm import tqdm
from utils import Config
import gc
from IPython import get_ipython
import numpy as np
import tensorflow as tf

frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
image_preprocess = Preprocess(frcnn_cfg)

lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
lxmert_base = LxmertForPreTraining.from_pretrained("unc-nlp/lxmert-base-uncased")

count, foil_accuracy, orig_passt, foil_detected, pairwise_acc_A, pairwise_acc_B, pairwise_acc_C = 0, 0, 0, 0, 0, 0, 0
nr_items, processing = 0, 0

# dirname = os.path.dirname
test_images = os.path.join(os.getcwd(), 'test_images')
#test_path = "/test_images/test_images/"


file = open('fresh_dict.json')
caption_dict = json.load(file)

caption_dict_0 = dict(list(caption_dict.items())[:len(caption_dict)//2])
caption_dict_1 = dict(list(caption_dict.items())[len(caption_dict)//2:])


# for item_id, info in tqdm(caption_dict_0.items()):
#     # print(item_id, info)
#     processing += 1
#     if info['foils'] != 'NOT AVAILABLE' and info['passive_foil'] != 'NOT AVAILABLE':
#         nr_items += 1
#         image_path = os.path.join(test_images, info['image_file'])
#         test_sentences = info['foils'], info['passive_foil']
#         images, sizes, scales_yx = image_preprocess(image_path)
#         output_dict = frcnn(
#             images, sizes, scales_yx = scales_yx, padding = 'max_detections', 
#             max_detections=frcnn_cfg.max_detections, return_tensors="pt"
#         )
#         # Very important that the boxes are normalized
#         normalized_boxes = output_dict.get("normalized_boxes")
#         features = output_dict.get("roi_features")
        
#         # run lxmert
#         # test_sentence = [test_sentence]

#         inputs = lxmert_tokenizer(
#             test_sentences,
#             padding="max_length",
#             max_length=30,  # 20
#             truncation=True,
#             return_token_type_ids=True,
#             return_attention_mask=True,
#             add_special_tokens=True,
#             return_tensors="pt"
#         )

#         # print('inputs are')
#         # print(inputs)

#         output_lxmert = lxmert_base(
#             input_ids=inputs.input_ids,
#             attention_mask=inputs.attention_mask,
#             visual_feats=features,
#             visual_pos=normalized_boxes,
#             token_type_ids=inputs.token_type_ids,
#             return_dict=True,
#             output_attentions=False,
#         )

#         m = torch.nn.Softmax(dim=1)
#         output = m(output_lxmert['cross_relationship_score'])
#         #output = output_lxmert['cross_relationship_score']
#         cross_score = output_lxmert['cross_relationship_score']
        


#         info['lxmert'] = dict()
#         info['lxmert']['foil'] = dict()
#         info['lxmert']['foil'][0] = output[0,0].item()
#         info['lxmert']['foil'][1] = output[1,0].item()

#         #prob_active[info['image_file']] = output[0,1].item()
#         #prob_passive[info['image_file']] = output[1,1].item()

#         #diff = output[0,1].item() - output[1,1].item()
#         #statistics['per_item'][info['image_file']] = dict()
#         #statistics['per_item'][info['image_file']]['active_prob'] = output[0,1].item()
#         #statistics['per_item'][info['image_file']]['passive_prob'] = output[1,1].item()
#         #statistics['per_item'][info['image_file']]['difference'] = diff

        
#         #info['lxmert'] = {'caption': 0, 'passive': 0} # 0 is not detected, 1 is detected
#         #info['lxmert']['caption'] = output[0, 1].item() # probability of fitting should be close to 1 for captions
#         #info['lxmert']['passive'] = output[1, 0].item() # probability of fitting, should be close to 0 for foils

#         if cross_score[1, 0] == cross_score[1, 1]:  # then something is wrong with the tokenisation
#             print(cross_score, test_sentences, inputs.input_ids)
#         else:
#             if cross_score[0, 0] < cross_score[0, 1]:  # the caption fits the image well
#                 foil_accuracy += 1
#                 orig_passt += 1
#             if cross_score[1, 0] >= cross_score[1, 1]:
#                 foil_detected += 1
#                 foil_accuracy += 1
#             if output[0, 1].item() > 0.5 and output[1, 1].item() > 0.5:
#                 if abs(output[0, 1].item() - output[1, 1].item()) < 0.1:
#                     pairwise_acc_A += 1
#                 elif abs(output[0, 1].item() - output[1, 1].item()) < 0.25:
#                     pairwise_acc_B += 1
#                 else:
#                     pairwise_acc_C += 1

#             count += 1
#         print('Item dealt with: ', processing, '   Skipped: ', processing - nr_items)
#         del output_lxmert
#         del output_dict
#         del normalized_boxes
#         del features

    
    
# print(f"""{count}/{nr_items}.
# FOIL det accuracy (Iacer): {foil_accuracy/count*50:.2f},
# Caption fits p_c: {orig_passt/count*100:.2f},
# FOIL detected p_f: {foil_detected/count*100:.2f},
# Pairwise accuracy acc_r class A: {pairwise_acc_A/count*100:.2f}
# Pairwise accuracy acc_r class B: {pairwise_acc_B/count*100:.2f}
# Pairwise accuracy acc_r class C: {pairwise_acc_C/count*100:.2f}"""
#)

core = 'passive'

try:
    # Change the current working Directory   
    dump_path = os.path.join(os.getcwd(), "/lxmert_results")
    os.chdir(dump_path)
    print("Directory changed")
except OSError:
    print("Can't change the Current Working Directory")

# with open(f'foil_og_1.json', 'w') as outfile:
#     json.dump(caption_dict, outfile, indent=4)

# with open(f'prob_active.json', 'w') as outfile:
#     json.dump(prob_active, outfile, indent=4)

# with open(f'prob_passive.json', 'w') as outfile:
#     json.dump(prob_passive, outfile, indent=4)

# with open(f'statistics_to_be_completed.json', 'w') as outfile:
#     json.dump(statistics, outfile, indent=4)

#### 2nd run

objects = dir()

for obj in objects:
  if not obj.startswith("__"):
    del globals()[obj]

from IPython.display import clear_output, Image, display
import torch
from processing_image import Preprocess
from modeling_frcnn import GeneralizedRCNN
from transformers import LxmertForQuestionAnswering, LxmertTokenizer, LxmertForPreTraining
import os, json
import random
from tqdm import tqdm
from utils import Config
import gc
from IPython import get_ipython


frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
image_preprocess = Preprocess(frcnn_cfg)

lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
lxmert_base = LxmertForPreTraining.from_pretrained("unc-nlp/lxmert-base-uncased")

count, foil_accuracy, orig_passt, foil_detected, pairwise_acc_A, pairwise_acc_B, pairwise_acc_C = 0, 0, 0, 0, 0, 0, 0
nr_items, processing = 0, 0

# dirname = os.path.dirname
test_images = os.path.join(os.getcwd(), 'test_images')
#test_path = "/test_images/test_images/"

prob_active = dict()
prob_passive = dict()

statistics = dict()
statistics['overview'] = dict()
statistics['per_item'] = dict()

file = open('fresh_dict.json')
caption_dict = json.load(file)

caption_dict_0 = dict(list(caption_dict.items())[:len(caption_dict)//2])
caption_dict_1 = dict(list(caption_dict.items()))

i = 0
for item_id, info in tqdm(caption_dict_1.items()):
    processing += 1
    if i >259 and i < 521:
        if info['foils'] != 'NOT AVAILABLE' and info['passive_foil'] != 'NOT AVAILABLE':
            nr_items += 1
            image_path = os.path.join(test_images, info['image_file'])
            test_sentences = [info['foils'], info['passive_foil']]
            images, sizes, scales_yx = image_preprocess(image_path)
            output_dict = frcnn(
                images, sizes, scales_yx = scales_yx, padding = 'max_detections', 
                max_detections=frcnn_cfg.max_detections, return_tensors="pt"
            )
            # Very important that the boxes are normalized
            normalized_boxes = output_dict.get("normalized_boxes")
            features = output_dict.get("roi_features")
            
            # run lxmert
            # test_sentence = [test_sentence]

            inputs = lxmert_tokenizer(
                test_sentences,
                padding="max_length",
                max_length=30,  # 20
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt"
            )

            output_lxmert = lxmert_base(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                visual_feats=features,
                visual_pos=normalized_boxes,
                token_type_ids=inputs.token_type_ids,
                return_dict=True,
                output_attentions=False,
            )

            m = torch.nn.Softmax(dim=1)
            output = m(output_lxmert['cross_relationship_score'])
            #output = output_lxmert['cross_relationship_score']
            cross_score = output_lxmert['cross_relationship_score']

            # for key in output_lxmert.keys():
            #     print('key: ', key)
            #     print('value: ', output_lxmert[key])

            info['lxmert'] = dict()
            info['lxmert']['foil'] = dict()
            info['lxmert']['foil'][0] = output[0,0].item()
            info['lxmert']['foil'][1] = output[1,0].item()

            # prob_active[info['image_file']] = output[0,1].item()
            # prob_passive[info['image_file']] = output[1,1].item()

            # diff = output[0,1].item() - output[1,1].item()
            # statistics['per_item'][info['image_file']] = dict()
            # statistics['per_item'][info['image_file']]['active_prob'] = output[0,1].item()
            # statistics['per_item'][info['image_file']]['passive_prob'] = output[1,1].item()
            # statistics['per_item'][info['image_file']]['difference'] = diff

            
            #info['lxmert'] = {'caption': 0, 'passive': 0} # 0 is not detected, 1 is detected
            #info['lxmert']['caption'] = output[0, 1].item() # probability of fitting should be close to 1 for captions
            #info['lxmert']['passive'] = output[1, 0].item() # probability of fitting, should be close to 0 for foils

            if cross_score[1, 0] == cross_score[1, 1]:  # then something is wrong with the tokenisation
                print(cross_score, test_sentences, inputs.input_ids)
            else:
                if cross_score[0, 0] < cross_score[0, 1]:  # the caption fits the image well
                    foil_accuracy += 1
                    orig_passt += 1
                if cross_score[1, 0] >= cross_score[1, 1]:
                    foil_detected += 1
                    foil_accuracy += 1
                if output[0, 1].item() > 0.5 and output[1, 1].item() > 0.5:
                    if abs(output[0, 1].item() - output[1, 1].item()) < 0.1:
                        pairwise_acc_A += 1
                    elif abs(output[0, 1].item() - output[1, 1].item()) < 0.25:
                        pairwise_acc_B += 1
                    else:
                        pairwise_acc_C += 1

                count += 1
            print('Item dealt with: ', processing, '   Skipped: ', processing - nr_items)
            
        if info['caption'] != 'NOT AVAILABLE' and info['passive'] != 'NOT AVAILABLE':
            nr_items += 1
            image_path = os.path.join(test_images, info['image_file'])
            test_sentences = info['caption'], info['passive']
            images, sizes, scales_yx = image_preprocess(image_path)
            output_dict = frcnn(
                images, sizes, scales_yx = scales_yx, padding = 'max_detections', 
                max_detections=frcnn_cfg.max_detections, return_tensors="pt"
            )
            # Very important that the boxes are normalized
            normalized_boxes = output_dict.get("normalized_boxes")
            features = output_dict.get("roi_features")
            
            # run lxmert
            # test_sentence = [test_sentence]

            inputs = lxmert_tokenizer(
                test_sentences,
                padding="max_length",
                max_length=30,  # 20
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt"
            )

            # print('inputs are')
            # print(inputs)

            output_lxmert = lxmert_base(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                visual_feats=features,
                visual_pos=normalized_boxes,
                token_type_ids=inputs.token_type_ids,
                return_dict=True,
                output_attentions=False,
            )

            m = torch.nn.Softmax(dim=1)
            output = m(output_lxmert['cross_relationship_score'])
            #output = output_lxmert['cross_relationship_score']
            cross_score = output_lxmert['cross_relationship_score']
            


            if info.get('lxmert', False) == False:
                info['lxmert'] = dict()
            info['lxmert']['caption'] = dict()
            info['lxmert']['caption'][0] = output[0,0].item()
            info['lxmert']['caption'][1] = output[1,0].item()
        
    i += 1


# print(f"""{count}/{nr_items}.
# FOIL det accuracy (Iacer): {foil_accuracy/count*50:.2f},
# Caption fits p_c: {orig_passt/count*100:.2f},
# FOIL detected p_f: {foil_detected/count*100:.2f},
# Pairwise accuracy acc_r class A: {pairwise_acc_A/count*100:.2f}
# Pairwise accuracy acc_r class B: {pairwise_acc_B/count*100:.2f}
# Pairwise accuracy acc_r class C: {pairwise_acc_C/count*100:.2f}"""
#)


with open(f'missing_vals_lxmert_pc.json', 'w') as outfile:
    json.dump(caption_dict, outfile, indent=4)

# with open(f'prob_active_1.json', 'w') as outfile:
#     json.dump(prob_active, outfile, indent=4)

# with open(f'prob_passive_1.json', 'w') as outfile:
#     json.dump(prob_passive, outfile, indent=4)

# with open(f'statistics_to_be_completed_1.json', 'w') as outfile:
#     json.dump(statistics, outfile, indent=4)