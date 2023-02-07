import spacy
from inv_pronoun import nouninv
from verb_inflector import verbPassifier

nlp = spacy.load('en_core_web_sm')


# noinspection PyUnusedLocal
def act2pass(doc, rec=False, verbose=False) -> str:
    """ 
    Sentences in the active voice that meet the requirements are made passive.
    The requirements are:   - sentence must be of the form SvO
                            - tense must be present simple or present continuous
    
    Note it works only for simple sentences (see test.py).

    Algorithm follows the same structure as passive-to-active parser from DanManN
    """

    parse = nlp(doc)
    newdoc = ''
    for sent in parse.sents:

        # Init parts of sentence to capture:
        subj = ''
        verb = ''
        verb_text = ''
        adverb = {'bef':'', 'aft':''}
        part = ''
        prep1 = '' # for pcomp
        aplural_d = False
        advcltree = None
        aux = ''
        xcomp = ''
        punc = '.'

        # novel parts
        coord = ''
        dobj = ''
        advcl = ''
        acomp = ''
        pobj = ''
        place_bef_o = False
        prep2 = '' # for compounds
        prep_verbmod = ''
        aplural_p = False
        prep0 = ''
        pobj_part = ''

        advcl_to_go = 0
        skip_some_toks = False
        mark = False



        processed = [False for _ in range(len(sent))]

        # Analyse dependency tree:
        for word in sent:
            if not processed[word.i]:
                if skip_some_toks and advcl_to_go > 0:
                    advcl_to_go -= 1
                    processed[word.i] = True
                    continue
                if word.dep_ == 'mark':
                    mark = True
                    if word.head.dep_ == 'advcl':
                        if word.head.head.dep_ in ('ROOT', 'auxpass'):
                            advcltree = word.head.subtree
                            advcl = word.head
                            processed[word.i] = True
                            number = len(list(advcltree)) - 1
                            advcl_to_go = number # skip the rest of the subsentence
                            skip_some_toks = True
                            continue
                if word.dep_ == 'advcl' and not mark:
                    advcl = word
                    for w in word.subtree:
                        processed[w.i] = True
                if word.dep_ == 'nsubj':
                    if not subj:
                        subj = ''.join(w.text_with_ws.lower() \
                                           if w.tag_ not in ('NNP','NNPS') else w.text_with_ws \
                                       for w in word.subtree).strip()
                        for w in word.subtree:
                            processed[w.i] = True
                if word.dep_ in ('advmod','npadvmod','oprd'):
                    if word.head.dep_ == 'ROOT':
                        if not verb:
                            adverb['bef'] = ''.join(w.text_with_ws.lower() \
                                                        if w.tag_ not in ('NNP','NNPS') else w.text_with_ws \
                                                    for w in word.subtree).strip()
                            for w in word.subtree:
                                processed[w.i] = True
                        else:
                            adverb['aft'] = ''.join(w.text_with_ws.lower() \
                                                        if w.tag_ not in ('NNP','NNPS') else w.text_with_ws \
                                                    for w in word.subtree).strip()
                            for w in word.subtree:
                                processed[w.i] = True
                if word.dep_ in ('aux','auxpass','neg'):
                    if word.head.dep_ == 'ROOT':
                        aux = word.text
                        for w in word.subtree:
                            processed[w.i] = True
                if word.dep_ == 'ROOT':
                    verb = word
                    verb_text = word.text
                    verb_pos = word.i
                    processed[word.i] = True
                if word.dep_ == 'prt':
                    if word.head.dep_ == 'ROOT':
                        part = ''.join(w.text_with_ws.lower() \
                                           if w.tag_ not in ('NNP','NNPS') else w.text_with_ws \
                                       for w in word.subtree).strip()
                        for w in word.subtree:
                            processed[w.i] = True
                if word.dep_ == 'acomp':
                    acomp = ''.join(w.text_with_ws.lower() \
                                        if w.tag_ not in ('NNP','NNPS') else w.text_with_ws \
                                    for w in word.subtree).strip()
                    for w in word.subtree:
                        processed[w.i] = True
                if word.dep_ == 'prep':
                    #if word.head.dep_ == 'ROOT':
                    #    prep1 = ''.join(w.text_with_ws.lower() \
                    #    if w.tag_ not in ('NNP','NNPS') else w.text_with_ws \
                    #    for w in word.subtree).strip()
                    prep_dep_list = list(w.dep_ for w in word.subtree)
                    if 'pobj' in prep_dep_list:
                        if not isinstance(verb, str) and sent[word.i-1] == verb:
                            prep_verbmod = word.text
                            pobj = ''.join(w.text_with_ws.lower() \
                                               if w.tag_ not in ('NNP','NNPS') and w.i != word.i else w.text_with_ws \
                                           for w in word.subtree if w.i != word.i).strip()
                            for w in word.subtree:
                                processed[w.i] = True
                        elif verb and (sent[word.i-1].dep_.endswith('obj') or sent[word.i-2].dep_.endswith('obj')):
                            dobj += ' ' + ''.join(w.text_with_ws.lower() \
                                                      if w.tag_ not in ('NNP','NNPS') else w.text_with_ws \
                                                  for w in word.subtree).strip()
                            for w in word.subtree:
                                processed[w.i] = True
                        else:
                            prep0 = ''.join(w.text_with_ws.lower() \
                                                if w.tag_ not in ('NNP','NNPS') else w.text_with_ws \
                                            for w in word.subtree).strip()
                            for w in word.subtree:
                                processed[w.i] = True
                    elif 'pcomp' in prep_dep_list:
                        prep1 = ''.join(w.text_with_ws.lower() \
                                            if w.tag_ not in ('NNP','NNPS') else w.text_with_ws \
                                        for w in word.subtree).strip()
                        for w in word.subtree:
                            processed[w.i] = True
                    elif 'compound' in prep_dep_list:
                        prep2 = ''.join(w.text_with_ws.lower() \
                                            if w.tag_ not in ('NNP','NNPS') else w.text_with_ws \
                                        for w in word.subtree).strip()
                        for w in word.subtree:
                            processed[w.i] = True
                if word.dep_.endswith('obj'):
                    if word.head.head.dep_ == 'ROOT':
                        if word.dep_ == 'dobj':
                            if not subj:
                                break
                            dobj = ''.join(w.text + ', ' \
                                               if w.dep_=='appos' else (w.text_with_ws.lower()
                                                                        if w.tag_ not in ('NNP','NNPS')
                                                                        else w.text_with_ws) \
                                           for w in word.subtree).strip()
                            aplural_d = word.tag_ in ('NNS','NNPS')
                            tree_list = list(w for w in word.subtree)
                            skip_first = True
                            for tok in tree_list:
                                if tok.dep_ == 'prep':
                                    if skip_first:
                                        skip_first = False
                                        continue
                                    pobj_part = ''.join(w.text_with_ws.lower() \
                                                            if w.tag_ not in ('NNP','NNPS') else w.text_with_ws \
                                                        for w in tok.subtree).strip()
                                    dobj_copy = dobj
                                    for i in range(len(pobj_part)):
                                        if pobj_part[- (1 + i)] != dobj_copy[- (1 + i)]:
                                            break
                                        dobj = dobj_copy[:-len(pobj_part)-1]
                                if tok.dep_ == 'acl':
                                    dobj += ','
                        if word.dep_ == 'pobj':
                            if word.head.dep_ == 'ROOT':
                                pobj = ''.join(w.text + ', ' \
                                                   if w.dep_=='appos' \
                                                   else (w.text_with_ws.lower()
                                                         if w.tag_ not in ('NNP','NNPS')
                                                         else w.text_with_ws) \
                                               for w in word.subtree).strip()
                                aplural_p = word.tag_ in ('NNS','NNPS')
                                if word.head.dep_ == 'dative':
                                    pobj = word.head.text + ' ' + pobj
                        if word.head.dep_ == 'prep':
                            place_bef_o = True
                        for w in word.subtree:
                            processed[w.i] = True
                if word.dep_ in ('xcomp','ccomp'):
                    if word.head.dep_ == 'ROOT':
                        xcomp = ''.join(w.text_with_ws.lower() \
                                            if w.tag_ not in ('NNP','NNPS') else w.text_with_ws \
                                        for w in word.subtree).strip()
                        that = xcomp.startswith('that')
                        xcomp = act2pass(xcomp, True).strip(' .')
                        if not xcomp.startswith('that') and that:
                            xcomp = 'that '+xcomp
                if word.dep_ == 'punct' and not rec:
                    if word.text != '"':
                        punc = word.text
                    processed[word.i] = True
                if word.dep_ == 'cc':
                    coord = word.text
                    if sent[word.i+1].head.tag_ in ('VB', 'VBD','VBG','VBN','VBP','VBZ','VERB') and \
                            sent[word.i+1].head.dep_ != 'ROOT':
                        coord += ' ' + ''.join(w.text_with_ws.lower() \
                                                   if w.tag_ not in ('NNP','NNPS') else w.text_with_ws \
                                               for w in sent[word.i+1].head.subtree).strip(' .')
                    for w in word.subtree:
                        processed[w.i] = True


        # exit if no SvO:
        if subj == '' and (dobj == '' or pobj == ''):
            newdoc += str(sent) + ' '
            if not rec and verbose:
                print('Warning: This sentence cannot be made passive.')
            return str(sent)


        # change tense of the verbs:
        if dobj:
            passive_voice = verbPassifier(verb, aux, aplural_d)
        else:
            passive_voice = verbPassifier(verb, aux, aplural_p)
        aux = passive_voice[0]
        verb = passive_voice[1]

        # invert nouns:
        subj = nouninv(subj)
        dobj = nouninv(dobj)

        advcl_pass = ''
        if advcl:
            marker = ''
            if list(advcl.subtree)[0].dep_ == 'mark':
                marker = list(advcl.subtree)[0].text.lower() + ' '
                temp_advcl = ''.join(w.text_with_ws.lower() \
                                         if w.tag_ not in ('NNP','NNPS') else w.text_with_ws \
                                     for w in advcl.subtree).strip()
                advcl = ''
                for i in range(len(temp_advcl)):
                    if i >= len(marker):
                        advcl += temp_advcl[i]
            else:
                advcl = ''.join(w.text_with_ws.lower() \
                                    if w.tag_ not in ('NNP','NNPS') else w.text_with_ws \
                                for w in advcl.subtree).strip()
            check_on_svo = nlp(advcl)
            contains_s = False
            contains_o = False
            for s in check_on_svo.sents:
                for w in s:
                    if w.dep_ == 'nsubj':
                        contains_s = True
                    if w.dep_.endswith('obj'):
                        contains_o = True
            if contains_s and contains_o:
                advcl_pass = ', ' + marker + act2pass(advcl, True).strip(' .')
            else:
                advcl_pass = ', ' + marker + advcl

        pobj_front = ''
        if place_bef_o or not dobj:
            pobj_front = pobj
            pobj = ''


        newsent0 = ' '.join(list(filter(None, [prep0,pobj_front,dobj,adverb['bef'],aux,verb,
                                               prep_verbmod,part,acomp,'by',subj,pobj_part,prep2,pobj,adverb['aft'],
                                               advcl_pass,coord,xcomp,prep1])))+punc
        newsent = ''


        # remove punctuation artefacts
        for i in range(len(newsent0)):
            if i == len(newsent0) - 1:
                newsent += newsent0[i]
            elif newsent0[i] == ' ' and newsent0[i+1] == ',':
                newsent += ''
            elif newsent0[i] == ',' and newsent0[i+1] == '.':
                newsent += ''
            else:
                newsent += newsent0[i]

        if not rec:
            newsent = newsent[0].upper() + newsent[1:]
        newdoc += newsent + ' '
    return newdoc
