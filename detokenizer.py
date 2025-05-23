from sacremoses import MosesDetokenizer
import sys

fname = sys.argv[1]

outputs = []
with open(f'{fname}','r') as infile:
    for line in infile:
        line = line.strip().split()
        outputs.append(line)

detok = MosesDetokenizer()
detoks = []
for output in outputs:
    detok_text = detok.detokenize(output)
    detoks.append(detok_text.strip())

fw = open(f'{fname}.detok','w')
for text in detoks:
    fw.write(text.strip()+'\n')
fw.close()

