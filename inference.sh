#model=checkpoints/ende-label-smoothing-transformer/checkpoint.best_bleu_29.0802.pt
#model=checkpoints/ende-label-smoothing-transformer/avg.pt
#output=outputs/base.txt
#model=checkpoints/ende-base-transformer/checkpoint.best_bleu_29.1402.pt
#model=checkpoints/ende-base-transformer/avg-epoch.pt
#output=outputs/als.txt
model=checkpoints/ende-mto-als-transformer/avg-best.pt
output=outputs/mto.txt
echo ${model} ${output}
fairseq-generate data-bin/wmt17_en_de --path ${model} --remove-bpe --beam 1 --batch-size 32  | grep ^H- | sort -V | cut -f3- > ${output}
