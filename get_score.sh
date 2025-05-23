output=mto.txt.detok
#output=base.txt.detok

sacrebleu outputs/test.de.detok -i outputs/$output -m bleu chrf

comet-score -s outputs/test.en.detok -t outputs/$output -r outputs/test.de.detok --quiet --only_system
comet-score -s outputs/test.en.detok -t outputs/$output --model Unbabel/wmt22-cometkiwi-da --quiet --only_system

