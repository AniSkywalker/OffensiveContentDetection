import csv
import re
# 0 hate speech, 1 offensive language, 2 neither
def convert_one_line(text):
    token = re.split(' |\n|\r', text)
    return ' '.join([t.strip() for t in token])

with open('hate_speech.train', newline='') as csvfile:
    with open('neither_hate_speech_format.train', 'w') as speech_writer:
        hate_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in hate_reader:
            if row[5] == "0":
                speech_writer.write('ID'+'\t' +  '1' + '\t' + convert_one_line(row[6]) + '\n')
            elif row[5] == "1":
                pass
            elif row[5] == "2":
                speech_writer.write('ID'+'\t' +  '0' + '\t' + convert_one_line(row[6]) + '\n')

