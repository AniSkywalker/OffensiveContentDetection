import csv
import re
# 0 hate speech, 1 offensive language, 2 neither
def convert_one_line(text):
    token = re.split(' |\n|\r', text)
    return ' '.join([t.strip() for t in token])
#
# with open('hate_speech.train', newline='') as csvfile:
#     with open('neither_hate_speech_format.train', 'w') as speech_writer:
#         hate_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
#         for row in hate_reader:
#             if row[5] == "0":
#                 speech_writer.write('ID'+'\t' +  '1' + '\t' + convert_one_line(row[6]) + '\n')
#             elif row[5] == "1":
#                 pass
#             elif row[5] == "2":
#                 speech_writer.write('ID'+'\t' +  '0' + '\t' + convert_one_line(row[6]) + '\n')
#

# with open('text_emotion.csv', newline='') as srcfile:
#     with open('text_emotion_processed.txt', 'w') as speech_writer:
#         text_reader = csv.reader(srcfile, delimiter=',', quotechar='"')
#         for row in text_reader:
#             if row[1] == "empty":
#                 speech_writer.write('ID'+'\t' +  '0' + '\t' + convert_one_line(row[3]) + '\n')
#             elif row[1] == "neutral":
#                 speech_writer.write('ID'+'\t' +  '1' + '\t' + convert_one_line(row[3]) + '\n')
#             elif row[1] == "happiness":
#                 speech_writer.write('ID'+'\t' +  '2' + '\t' + convert_one_line(row[3]) + '\n')
#             elif row[1] == "worry":
#                 speech_writer.write('ID'+'\t' +  '3' + '\t' + convert_one_line(row[3]) + '\n')
#             elif row[1] == "love":
#                 speech_writer.write('ID'+'\t' +  '4' + '\t' + convert_one_line(row[3]) + '\n')
#             elif row[1] == "enthusiasm":
#                 speech_writer.write('ID'+'\t' +  '5' + '\t' + convert_one_line(row[3]) + '\n')
#             elif row[1] == "fun":
#                 speech_writer.write('ID'+'\t' +  '6' + '\t' + convert_one_line(row[3]) + '\n')
#             elif row[1] == "hate":
#                 speech_writer.write('ID'+'\t' +  '7' + '\t' + convert_one_line(row[3]) + '\n')
#             elif row[1] == "relief":
#                 speech_writer.write('ID'+'\t' +  '8' + '\t' + convert_one_line(row[3]) + '\n')
#             elif row[1] == "sadness":
#                 speech_writer.write('ID'+'\t' +  '9' + '\t' + convert_one_line(row[3]) + '\n')

from collections import defaultdict
import xml.etree.ElementTree as ET
f = open('affectivetext_test.emotions.gold', 'r')
data = f.readlines()
id_score = defaultdict()
for line in data:
    token = line.strip().split(' ')
    if token[0] not in id_score:
        id_score[token[0]] = token[1:]

def more_than_50(element):
    return element >= 50

w = open('affective.txt', 'w+')
tree = ET.parse('affectivetext_test.xml')
root = tree.getroot()
for child in root:
    if child.attrib['id'] in id_score:
        scores = id_score.get(child.attrib['id'])
        for i, score in enumerate(scores):
            if int(score) > 50:
                w.write('ID' + '\t' + str(i) + '\t' + child.text + '\n')
    # figures = line.split(' ')
    #     w.write('ID' + '\t' + figures[1] + '\t' + figures[2] + '\t' + figures[3] + '\t' + figures[4] + '\t' + figures[5]
    #             + '\t' + figures[6].strip() + '\t' + child.text)