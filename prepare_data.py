import os
import csv
import collections
import matplotlib.pyplot as plt
import numpy as np

data_root = './gmb-2.2.0/data/'

fnames = []
for root, dirs, files in os.walk(data_root):
    for filename in files:
        if filename.endswith(".tags"):
            fnames.append(os.path.join(root, filename))

print(fnames[:2])

ner_tags = collections.Counter()
iob_tags = collections.Counter()


def strip_ner_subcat(tag):
    # NER tags are of form {cat}-{subcat}
    # eg tim-dow. We only want first part
    return tag.split("-")[0]


def iob_format(ners):
    # converts IO tags into BIO format
    # input is a sequence of IO NER tokens
    # convert this: O, PERSON, PERSON, O, O, LOCATION, O
    # into: O, B-PERSON, I-PERSON, O, O, B-LOCATION, O
    iob_tokens = []
    for idx, token in enumerate(ners):
        if token != 'O':  # !other
            if idx == 0:
                token = "B-" + token  # start of sentence
            elif ners[idx - 1] == token:
                token = "I-" + token  # continues
            else:
                token = "B-" + token
        iob_tokens.append(token)
        iob_tags[token] += 1
    return iob_tokens


total_sentences = 0
outfiles = []
for idx, file in enumerate(fnames):
    with open(file, 'rb') as content:
        data = content.read().decode('utf-8').strip()
        sentences = data.split("\n\n")
        print(idx, file, len(sentences))
        total_sentences += len(sentences)

        with open("./ner/" + str(idx) + "-" + os.path.basename(file), 'w') as outfile:
            outfiles.append("./ner/" + str(idx) + "-" + os.path.basename(file))
            writer = csv.writer(outfile)

            for sentence in sentences:
                toks = sentence.split('\n')
                words, pos, ner = [], [], []

                for tok in toks:
                    t = tok.split("\t")
                    words.append(t[0])
                    pos.append(t[1])
                    ner_tags[t[3]] += 1
                    ner.append(strip_ner_subcat(t[3]))
                writer.writerow([" ".join(words),
                                 " ".join(iob_format(ner)),
                                 " ".join(pos)])

print("total number of sentences: ", total_sentences)

print(ner_tags)
print(iob_tags)

labels, values = zip(*iob_tags.items())

indexes = np.arange(len(labels))

plt.bar(indexes, values)
plt.xticks(indexes, labels, rotation='vertical')
plt.margins(0.01)
plt.subplots_adjust(bottom=0.15)
plt.show()
