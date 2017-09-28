
from random import choice, shuffle
from sys import argv

def word_map(text):
    ''' Maps each word to a list of all words that follow it. '''
    D = {w : [] for w in text}; D[''] = text
    [D[w].append(text[i+1]) for i, w in enumerate(text) if i < len(text)-1]
    return D

def print_mimic(D, w):
    ''' Recursively prints random words from a word map. '''
    print w,
    try: print_mimic(D, choice(D[w]))
    except: exit()

def main():
    with open(argv[1], 'r') as infile:
        text1 = infile.read().split()
    with open(argv[2], 'r') as infile:
        text2 = infile.read().split()
    text = text1+text2
    shuffle(text)

    D = word_map(text)
    print_mimic(D, '')

if __name__ == '__main__':
    main()
