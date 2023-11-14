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
    del file_to_timelist['basename_file'] # remove dict entry of header
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
    try:
        # Open and read the structured data file
        with open(filename, 'r', encoding='ISO-8859-1') as file:
            first_verse_found = False
        
            for line in file:
                if not line.strip():
                    continue
                # Split the line into columns based on tab ('\t') separator
                columns = line.strip().split('\t')

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
    except(FileNotFoundError):
        print(f"No such file or directory {filename}")
    
    return concatenated_lines

def split_text_stanza(filename):
    # Initialize variables to store the concatenated lines and the current line
    concatenated_lines = []
    current_line = []
    try:
        # Open and read the structured data file
        with open(filename, 'r', encoding='ISO-8859-1') as file:
            first_verse_found = False
        
            for line in file:
                if not line.strip():
                    continue
                # Split the line into columns based on tab ('\t') separator
                columns = line.strip().split('\t')

                # Check if column 7 is not "S"
                if columns[6] != "S" and columns[6] != "O":
                    # Append the token from column 0 to the current line
                    current_line.append(columns[0])
                    if columns[6] == "V":
                        current_line.append("^")

                # If column 7 is "V" or "S", start a new line
                elif columns[6] == "S" or columns[6] == "O":
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
    except(FileNotFoundError):
        print(f"No such file or directory {filename}")
    
    return concatenated_lines

def text_to_transcript(f, indir, outdir):
    concatenated_lines = split_text_stanza(f"{indir}/{f}.token")
    try:
        with open(f"{outdir}/{f}/transcript.txt", "w", encoding="utf-8") as out:
            for i,line in enumerate(concatenated_lines):
                line = f"{f}_{i}\t{line}\n"
                # print(line)
                out.write(line)
    except():
        print(f"Couldn't create file {out}")

def read_filelist(filelist):
    files = []
    with open(filelist, "r") as fl:
        for line in fl:
            files.append(line.rstrip())
    return files

if __name__ == '__main__':
    problematic_files_verses = ["tts-dd-1158-m01-s01-t01-v01",
                         "tts-dd-433-m01-s01-t01-v01",
                        "tts-dd-433-m01-s01-t10-v01",
                        "tts-dd-433-m01-s01-t11-v01",
                        "tts-dd-433-m01-s01-t12-v01",
                        "tts-dd-433-m01-s01-t13-v02",
                        "tts-dd-433-m01-s01-t15-v01",
                        "tts-dd-433-m01-s01-t16-v01",
                        "tts-dd-433-m01-s01-t17-v01",
                        "tts-dd-433-m01-s01-t20-v01",
                        "tts-dd-433-m01-s01-t22-v01",
                        "tts-dd-433-m01-s01-t24-v01",
                        "tts-dd-1158-m01-s01-t03-v01",
                        "tts-dd-1158-m01-s01-t17-v01",
                        "tts-dd-1158-m01-s01-t18-v01",
                        "tts-dd-1158-m01-s01-t20-v01",
                        "tts-dd-1158-m01-s01-t21-v01",
                        "tts-dd-1158-m01-s01-t23-v01",
                        "tts-dd-1158-m01-s01-t30-v01",
                        "tts-dd-1158-m01-s01-t34-v01",
                        "tts-dd-1158-m01-s01-t42-v01",
                        "tts-dd-1158-m01-s01-t43-v01",
                        "tts-dd-1158-m01-s01-t46-v01",
                        "tts-dd-1158-m01-s01-t49-v01",
                        "tts-dd-1357-m01-s01-t02-v01",
                        "tts-dd-1357-m01-s01-t09-v02",
                        "tts-dd-2007-m01-s01-t20-v01",
                        "tts-dd-2367-m01-s01-t02-v01",
                        "tts-dd-2367-m01-s01-t04-v01",
                        "tts-dd-2367-m01-s01-t06-v01",
                        "tts-dd-2367-m01-s01-t09-v01",
                        "tts-dd-2367-m01-s01-t11-v01",
                        "tts-dd-2367-m01-s01-t13-v01",
                        "tts-dd-2367-m01-s01-t14-v01",
                        "tts-dd-2367-m01-s01-t15-v01",
                        "tts-dd-2367-m01-s01-t16-v01",
                        "tts-dd-2367-m01-s01-t17-v01",
                        "tts-dd-2367-m01-s01-t18-v01",
                        "tts-dd-2367-m01-s01-t28-v01",
                        "tts-dd-2367-m01-s01-t31-v01",
                        "tts-dd-2367-m01-s01-t33-v01",
                        "tts-dd-2367-m01-s01-t35-v01",
                        "tts-dd-2367-m01-s01-t36-v01",
                        "tts-dd-2367-m01-s01-t42-v01",
                        "tts-dd-2367-m01-s01-t43-v01",
                        "tts-dd-2367-m01-s01-t45-v01",
                        "tts-dd-2367-m01-s01-t49-v01",
                        "tts-dd-3976-m01-s01-t06-v01",
                        "tts-dd-3976-m01-s01-t07-v01",
                        "tts-dd-3976-m01-s01-t12-v01",
                        "tts-dd-3976-m01-s01-t13-v01",
                        "tts-dd-3976-m01-s01-t14-v01",
                        "tts-dd-3976-m01-s01-t15-v01",
                        "tts-dd-3976-m01-s01-t16-v01",
                        "tts-dd-3976-m01-s01-t17-v01",
                        "tts-dd-3976-m01-s01-t24-v01",
                        "tts-dd-3976-m02-s01-t06-v01",
                        "tts-dd-3976-m02-s01-t09-v01",
                        "tts-dd-3976-m02-s01-t14-v01",
                        "tts-dd-3976-m02-s01-t18-v01",
                        "tts-dd-3976-m02-s01-t19-v01",
                        "tts-dd-3976-m02-s01-t20-v01",
                        "tts-dd-3976-m02-s01-t23-v01",
                        "tts-dd-3976-m02-s01-t27-v01",
                        "tts-dd-3976-m02-s01-t29-v01",
                        "tts-dd-3976-m02-s01-t32-v01",
                        "tts-dd-3976-m02-s01-t33-v01",
                        "tts-dd-3976-m02-s01-t36-v01",
                        "tts-dd-3976-m02-s01-t40-v01",
                        "tts-dd-3976-m02-s01-t43-v01",
                        "tts-dd-3976-m02-s01-t52-v01",
                        "tts-dd-3976-m02-s01-t58-v01",
                        "tts-dd-3976-m02-s01-t63-v01"]
    
    # problematic_files_stanzas = ["tts-dd-433-m01-s01-t10-v01",
    #                             "tts-dd-433-m01-s01-t11-v01",
    #                             "tts-dd-433-m01-s01-t12-v01",
    #                             "tts-dd-433-m01-s01-t15-v01",
    #                             "tts-dd-433-m01-s01-t16-v01",
    #                             "tts-dd-433-m01-s01-t17-v01",
    #                             "tts-dd-433-m01-s01-t20-v01",
    #                             "tts-dd-433-m01-s01-t22-v01",
    #                             "tts-dd-433-m01-s01-t24-v01",
    #                             "tts-dd-1158-m01-s01-t01-v01",
    #                             "tts-dd-1158-m01-s01-t17-v01",
    #                             "tts-dd-1158-m01-s01-t18-v01",
    #                             "tts-dd-1158-m01-s01-t20-v01",
    #                             "tts-dd-1158-m01-s01-t21-v01",
    #                             "tts-dd-1158-m01-s01-t23-v01",
    #                             "tts-dd-1158-m01-s01-t30-v01",
    #                             "tts-dd-1158-m01-s01-t43-v01",
    #                             "tts-dd-1158-m01-s01-t46-v01",
    #                             "tts-dd-1158-m01-s01-t49-v01",
    #                             "tts-dd-2007-m01-s01-t20-v01",
    #                             "tts-dd-2367-m01-s01-t02-v01",
    #                             "tts-dd-2367-m01-s01-t04-v01",
    #                             "tts-dd-2367-m01-s01-t06-v01",
    #                             "tts-dd-2367-m01-s01-t09-v01",
    #                             "tts-dd-2367-m01-s01-t11-v01",
    #                             "tts-dd-2367-m01-s01-t13-v01",
    #                             "tts-dd-2367-m01-s01-t14-v01",
    #                             "tts-dd-2367-m01-s01-t16-v01",
    #                             "tts-dd-2367-m01-s01-t17-v01",
    #                             "tts-dd-3976-m01-s01-t03-v01",
    #                             "tts-dd-3976-m01-s01-t06-v01",
    #                             "tts-dd-3976-m01-s01-t07-v01",
    #                             "tts-dd-3976-m01-s01-t12-v01",
    #                             "tts-dd-3976-m01-s01-t13-v01",
    #                             "tts-dd-3976-m01-s01-t14-v01",
    #                             "tts-dd-3976-m01-s01-t15-v01",
    #                             "tts-dd-3976-m01-s01-t16-v01",
    #                             "tts-dd-3976-m01-s01-t17-v01",
    #                             "tts-dd-3976-m01-s01-t24-v01",
    #                             "tts-dd-3976-m02-s01-t06-v01",
    #                             "tts-dd-3976-m02-s01-t09-v01",
    #                             "tts-dd-3976-m02-s01-t14-v01",
    #                             "tts-dd-3976-m02-s01-t18-v01",
    #                             "tts-dd-3976-m02-s01-t19-v01",
    #                             "tts-dd-3976-m02-s01-t20-v01",
    #                             "tts-dd-3976-m02-s01-t23-v01",
    #                             "tts-dd-3976-m02-s01-t27-v01",
    #                             "tts-dd-3976-m02-s01-t29-v01",
    #                             "tts-dd-3976-m02-s01-t32-v01",
    #                             "tts-dd-3976-m02-s01-t33-v01",
    #                             "tts-dd-3976-m02-s01-t36-v01",
    #                             "tts-dd-3976-m02-s01-t40-v01",
    #                             "tts-dd-3976-m02-s01-t43-v01",
    #                             "tts-dd-3976-m02-s01-t52-v01",
    #                             "tts-dd-3976-m02-s01-t58-v01",
    #                             "tts-dd-3976-m02-s01-t63-v01"]

    problematic_files_stanzas = []

    # timedict = read("/mount/arbeitsdaten/textklang/data/timestamps_Strophe.txt")
    timedict = read("/mount/arbeitsdaten/textklang/synthesis/poetry_styles_toucan/IMS-Toucan/Strophen_timestamps_neu.csv")
    #print(timedict)
    root = "/mount/projekte/textklang/Audio-Pipeline/Data"
    # root = "/mount/projekte/textklang/omeka_s_export"
    out_dir = "/mount/arbeitsdaten/textklang/synthesis/Multispeaker_PoeticTTS_Data/TAI/Schiller"

    files = read_filelist("DSchiller_files.txt")
    print(len(files))
    print(files)

    for f, timepoints in timedict.items():
        if f in files:
            if f"{f}.wav" not in os.listdir("/mount/projekte/textklang/Audio-Pipeline/Data") or f"{f}.token" not in os.listdir("/mount/projekte/textklang/Audio-Pipeline/Data") :
                print(f"{f} not in Audio-Pipeline Data")
                continue
            print(f, " ", timepoints)
            cut_audio(f"{root}/{f}.wav", timepoints, out_dir)
            text_to_transcript(f, root, out_dir)



    

    