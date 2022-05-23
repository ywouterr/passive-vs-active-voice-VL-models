import spacy
import lemminflect
import numpy
from lemminflect import getInflection, getAllInflections, getAllInflectionsOOV
nlp = spacy.load('en_core_web_sm')

def verbPassifier(root, aux, plural):
    infl_root = getInflection(root.lemma_, 'VBN')[0]
    infl_aux = ''
    if not plural:
        infl_aux = getInflection('be', 'VBZ')[0]
    else:
        infl_aux = getInflection('be', 'VBP')[1]
    if aux:
        infl_aux += ' being'

    return infl_aux, infl_root