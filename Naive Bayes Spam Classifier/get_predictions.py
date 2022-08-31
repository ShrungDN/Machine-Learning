import pickle
import os
from NB import *

directory = 'test'
files = []
filepaths = []

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        if filename[:5] == 'email':
            files.append(filename)
            filepaths.append(f)

test_data = []

# for i in range(len(files)):
#     filepath = directory + '\\' + 'email' + f'{i+1}' + '.txt'

for f in filepaths:
    with open(f, 'r') as ftxt:
        lines = ftxt.read()
        lines = re.sub('ArialMT', '', lines)
        lines = re.sub('Helvetica', '', lines)
        lines = re.sub('\\\\[A-Za-z0-9]*','',lines)
        lines = re.sub('[{}]','',lines)
        lines = re.sub('\\n',' ', lines)
        lines = re.sub(' +', ' ', lines)
        lines = re.sub(';', '', lines)
        test_data.append(lines)

with open('NBmodel.pkl', 'rb') as f:
    NBmodel = pickle.load(f)

# predict = NBmodel.predict(test_data) # +1 = spam, 0 = not spam
# print(predict)

msg = ["The Biotech Research Club is back with the Alumni Talk series. This week's talk features Dr. Naga Sirisha Parimi, an alumnus of the Batch of 2010, who is a Research Biotechnologist in the R&D group at Cargill, Incorporated, a multinational food and agribusiness industry in the United States. Please find attached the details of the talk, the poster, and a query form in which you can ask any questions you might have for the speaker. We look forward to seeing you there! ", "A practical guide to use Mendeley and Zotero for Postgraduate students, researchers and academics in any discipline."]
predict = NBmodel.predict(msg) # +1 = spam, 0 = not spam
print(predict)

# predictions = {}
# for i in range(len(files)):
#     predictions[files[i]] = predict[i]
# print("The predictions are: \n", predictions)