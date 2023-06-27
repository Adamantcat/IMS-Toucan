import opensmile
import json
import audiofile
import os

def extract_features(audio):
    signal, sampling_rate = audiofile.read(audio, always_2d=True,)

    smile = opensmile.Smile(
    feature_set='/mount/arbeitsdaten/dialog-1/kochja/venvs/teamlab_venv/lib64/python3.10/site-packages/opensmile/core/config/egemaps/v02_mfcc/eGeMAPSv02.conf',
    feature_level='lld',)
    features_LLDs = smile.process_signal(signal,sampling_rate)

    return features_LLDs.to_dict('list')

def extract_all(input_dir, out_file):
    features_dict = {}
    for poem in os.listdir(os.path.join(input_dir)):
        print(poem)
        for wav in os.listdir(os.path.join(input_dir, poem)):
            print(wav)
            features_dict[wav] = {}
            features_dict[wav]['features'] = extract_features(os.path.join(input_dir, poem, wav))
            
    with open(out_file, 'w') as out:
        json.dump(features_dict, out, sort_keys=True, indent=4)

if __name__ == '__main__':
    extract_all("/mount/arbeitsdaten/textklang/synthesis/Poetry_Styles/verses", "poems_features_eGeMAPS_mfcc20.json")