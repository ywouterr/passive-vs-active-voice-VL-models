from a2p import *

text3 = 'She gives a well deserved raise.'
text = 'Everyone loves my cat Puck.'
text2 = 'The woman encourages the man, while he washes the dishes.'
sent = 'Grandma picks up some pens.'
sent2 = 'Puck is always looking on the bright side of life.'
sent3 = 'The man eats a sandwich.'
sent4 = 'I like big huts and I can not lie.'
sent5 = 'I am taking a picture.'
all_in = 'The woman sits on the couch.'
sent6 = 'While the sun is setting, Harold prepares his dinner.'
sent7 = 'Sam, my cousin, takes out a $ 10 million loan.'
sent8 = 'The lab assistent is preparing genetically modified food.'
sent9 = 'The child causes havoc as the night is falling.'
sent10 = 'The garbage man purchases the Ferrari without paying a premium.'
sent11 = 'I see a cat with a telescope.'
sent12 = 'Mary spends half of her salary on video games.'
sent13 = 'While the car is passing by, Sara sings a beautiful song, inspired by birds.'
sent14 = 'The wrecking ball slams the building, causing it to collapse.'
sent15 = 'The buzzing fly is making noises because flies have wings.'
sent16 = 'A rainbow crosses the sky.'
sent17 = 'On the right side a car overtakes a slug.'
sent18 = 'On the left side mom picks tomatoes from the garden.'
sent19 = 'The cook reads the book which he bought.'
sent20 = "The officer sends flowers to the best mother in the world."

sentX = 'A man shouts at a woman.'
sentY = 'On the right side a car overtakes a slug.'
sentZ = 'On the left side mom picks tomatoes from the garden.'
sentA = 'A cat sits on a rug.'

all_sentences = [text3, text, text2, sent, sent2, sent3, sent4, sent5, all_in, sent6, sent7, sent8, sent9, sent10,
                 sent11, sent12, sent13, sent14, sent15, sent16, sent17, sent18, sent19, sent20, sentX, sentY, sentZ,
                 sentA]

print(act2pass(sentA))

for token in nlp(sentA):
    print(token.text, token.dep_, token.tag_, token.head, list(token.subtree))
