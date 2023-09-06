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
    try:
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

    except(FileNotFoundError):
        print(f"No such file or directory: {filename}")
    
def split_text(filename):
    # Initialize variables to store the concatenated lines and the current line
    concatenated_lines = []
    current_line = []

    # Open and read the structured data file
    with open(filename, 'r', encoding='ISO-8859-1') as file:
        first_verse_found = False
       
        for line in file:
            if not line.strip():
                continue
            # Split the line into columns based on tab ('\t') separator
            columns = line.strip().split('\t')
            print(columns)

            # Authornames, Title, etc. go into the same line
            if columns[7] != "VERSE" and not first_verse_found:
                current_line.append(columns[0])

            elif columns[7] == "VERSE" and not first_verse_found:
                first_verse_found = True
                
                concatenated_line = ' '.join(current_line)
                if concatenated_line:
                    concatenated_lines.append(concatenated_line)
                # Start a new line
                current_line = []
                current_line.append(columns[0])

            # Check if column 7 is not "V" or "S"
            elif columns[6] not in ("V", "S"):
                # Append the token from column 1 to the current line
                current_line.append(columns[0])

            # If column 7 is "V" or "S", start a new line
            elif columns[6] == "V" or columns[6] == "S":
                # Concatenate the tokens in the current line and add it to the result
                current_line.append(columns[0])
                concatenated_line = ' '.join(current_line)
                if concatenated_line:
                    concatenated_lines.append(concatenated_line)

                # Start a new line
                current_line = []

    # Concatenate the last line (if any)
    concatenated_line = ' '.join(current_line)
    if concatenated_line:
        concatenated_lines.append(concatenated_line)

    # Join the concatenated lines into one string with newline separators
    # result = ' '.join(concatenated_lines)

    # Print or save the result as needed
    return concatenated_lines

def text_to_transcript(f, indir, outdir):
    concatenated_lines = split_text(f"{indir}/{f}.token")
    with open(f"{outdir}/{f}/transcript.txt", "w", encoding="utf-8") as out:
        for i,line in enumerate(concatenated_lines):
            line = f"{f}_{i}\t{line}\n"
            print(line)
            out.write(line)

if __name__ == '__main__':
    timedict = read("/mount/arbeitsdaten/textklang/data/timestamps_Vers_Titel.txt")
    print(timedict)
    root = "/projekte/textklang/Audio-Pipeline/Data"
    out_dir = "/mount/arbeitsdaten/textklang/synthesis/EACL_2024/Vers_no_Title"

    text_to_transcript("tts-dd-1158-m01-s01-t02-v01", root, out_dir)

    #for f, timepoints in timedict.items():
        # print(f, " ", timepoints)
     #   cut_audio(f"{root}/{f}.wav", timepoints, out_dir)



    

    