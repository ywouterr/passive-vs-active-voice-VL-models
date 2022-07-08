import spacy
import lemminflect
import numpy
from lemminflect import getInflection, getAllInflections, getAllInflectionsOOV
nlp = spacy.load('en_core_web_sm')

def verbPassifier(root, aux, plural):
    
    infl_aux = ''
    if not plural:
        infl_aux = getInflection('be', 'VBZ')[0]
    else:
        infl_aux = getInflection('be', 'VBP')[1]
    if aux:
        infl_aux += ' being'
        infl_root = getInflection(root.lemma_, 'VBN')[0]
    else:
        infl_root = getInflection(root.lemma_, 'VBG')[0]

    return infl_aux, infl_root

# present continuous
def conv2pc(root, aux, plural, I):
    infl_aux = ''
    if I:
        infl_aux = getInflection('be', 'VBP')[0]
    elif not plural:
        infl_aux = getInflection('be', 'VBZ')[0]
    else:
        infl_aux = getInflection('be', 'VBP')[1]
    if aux:
        infl_aux += ' being'
        infl_root = getInflection(root.lemma_, 'VBN')[0]
    else:
        infl_root = getInflection(root.lemma_, 'VBG')[0]
        
    return infl_aux, infl_root

# congruence verb with obj
def makeObjActive(root, aux, plural, I):
    infl_aux = ''
    if I:
        infl_aux = getInflection('be', 'VBP')[0]
    elif not plural:
        infl_aux = getInflection('be', 'VBZ')[0]
    else:
        infl_aux = getInflection('be', 'VBP')[1]
    if aux:
        infl_root = getInflection(root.lemma_, 'VBN')[0]
    else:
        infl_root = getInflection(root.lemma_, 'VBG')[0]
    
    return infl_aux, infl_root


# print(getAllInflections('sing'))
# print(getAllInflections('be'))
