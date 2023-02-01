from pydub import AudioSegment
import os

def read(filename):
    with open(filename, "r") as f:
        file_to_timelist = {}
        lines = f.readlines()
        for line in lines:
            name, time = line.split(",")
            time = time.replace("\n", "")
            if name in file_to_timelist:
                timepoints = file_to_timelist[name]
                if timepoints[-1] != time:
                    timepoints.append(time)
            else:
                file_to_timelist[name] = [time]
    del file_to_timelist['file'] # remove dict entry of header
    return file_to_timelist

def cut_audio(filename, timepoints, out_dir):
    poem = AudioSegment.from_file(filename, format="wav")
    poem_name = filename.split("/")[-1].replace(".wav", "")
    for i in range(len(timepoints)):
        if i == 0:
            start = 0.0
        else:
            start = float(timepoints[i-1]) * 1000
        end = float(timepoints[i]) * 1000
        verse = AudioSegment.silent(duration=50) + poem[start:end] + AudioSegment.silent(duration=50) # add 50 ms of silence at beginning and end of verse
        if not os.path.exists(f"{out_dir}/{poem_name}"):
            os.makedirs(f"{out_dir}/{poem_name}")
        
        file_handle = verse.export(f"{out_dir}/{poem_name}/{poem_name}_{i}.wav", format="wav")
    



if __name__ == '__main__':
    timedict = read("/mount/arbeitsdaten/textklang/data/timestamps_Strophe.txt")
    print(timedict)
    root = "/projekte/textklang/Audio-Pipeline/Data"
    out_dir = "/mount/arbeitsdaten/textklang/synthesis/styles/stanzas"
    for f, timepoints in timedict.items():
        print(f, " ", timepoints)
        cut_audio(f"{root}/{f}.wav", timepoints, out_dir)