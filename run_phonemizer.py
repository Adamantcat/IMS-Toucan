from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend

import re
import sys
import os
import json
import pandas as pd

import torch
from dragonmapper.transcriptions import pinyin_to_ipa
from phonemizer.backend import EspeakBackend
from pypinyin import pinyin

from Utility.storage_config import PREPROCESSING_DIR
from Preprocessing.articulatory_features import generate_feature_table
from Preprocessing.articulatory_features import get_feature_to_index_lookup
from Preprocessing.articulatory_features import get_phone_to_id

from flair.data import Sentence
from flair.models import SequenceTagger
import flair

import sys

from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend

tf = ArticulatoryCombinedTextFrontend(language="fr")
#tf.string_to_tensor("On ne le vit plus que par le dessous de sa nef et bientôt il disparut dans les profondeurs de l'azur.", view=True)
#tf.string_to_tensor('Nous allons rester plus longtemps, nous n’avons plus de travail! On veut plus de devoirs à l’école. J’aime le vin plus que la bière. C’était plus intéressant que je ne pensais. C’était le plus intéressant.', view=True)
#tf.string_to_tensor("je ne t'aime pas j'aime mon chat plus que toi.", view=True)
tf.string_to_tensor("Que prévoit la proposition de loi pour faciliter les adoptions?", view=True)


sys.exit(0)

lex = pd.read_csv("http://www.lexique.org/databases/Lexique382/Lexique382.tsv", sep='\t')

# separe les noms et les verbes dans deux dataframes:
# TODO: add other ambiguous pos tags
# ambiguous tags are: nom, nom de famille, adverbe, adjectif, préposition, verbe, nom propre
lex_noms = lex.loc[lex.cgram == 'NOM']
lex_verbs = lex.loc[lex.cgram == 'VER']

def disambiguate_nom(word, entry):
    if 'f' in entry['meta']:
        print('female')
        print(word, entry)
    # lex_entries = lex_noms.loc[lex_noms.ortho == word]
    # print(lex_entries)
        

with open("Preprocessing/french_homographs_preprocessed.json", "r", encoding="utf8") as f:
    homographs = json.loads(f.read())

ambiguous = list()
tagset = set()
multiple_pronunciations = list()
for word in homographs.keys():
    entries = homographs[word]
    
    for i,entry in enumerate(entries):
        pos = entry['pos']
        pronunciation = entry['pronunciation']
        tagset.add(pos)
        if type(entry['pronunciation']) == list:
            multiple_pronunciations.append((word, entry['pronunciation']))
        for other in entries[i+1:]: #loop through following entries of the same word to check if there are entries with same pos but different pronunciation
            if other['pos'] == pos and other['pronunciation'] == pronunciation: # same pos and same pronunciation, we can ignore them for now, maybe need to collapse into one entry
                # print('found same pos and same pronunciation: ')
                # print(word)
                # print(entry, " ", pos, " ", pronunciation)
                # print(other, " ", other['pos'], " ", other['pronunciation'])
                pass
            elif other['pos'] == pos and other['pronunciation'] != pronunciation: # same pos, different pronunciation, need to disambiguate
                #if pos == 'nom':
                ambiguous.append(word)
                    # print('found same pos but DIFFERENT pronunciation: ')
                    # print(word)
                    # print(entry, " ", pos, " ", pronunciation)
                    # print(other, " ", other['pos'], " ", other['pronunciation'])
                    #disambiguate_nom(word, entry)
                    # print()
print(tagset)
print(len(ambiguous))  
print("entries with multiple pronunciations: ", len(multiple_pronunciations))
for e in multiple_pronunciations:
    print(e)
# for e in ambiguous:
#     print(e)  
          
# print(len(homographs.keys()))
    