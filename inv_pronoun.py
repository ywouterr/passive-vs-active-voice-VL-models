noundict = {'i':'me', 'I':'me', 'we':'us', 'he':'him', 'she':'her', 'they':'them', 'them':'they', 'her':'she', 'him':'he', 'us':'we', 'me':'I'}

def nouninv(noun):
    if noun in noundict:
        return noundict[noun]
    return noun