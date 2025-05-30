d_model=1024
fnn_dim=$(echo "4 * ${d_model}" | bc -l)
num_head=16
lr=0.002
echo ${d_model} ${fnn_dim} ${lr}

fairseq-train data-bin/wmt17_en_de \
  --arch transformer \
  --encoder-embed-dim ${d_model} \
  --encoder-ffn-embed-dim ${fnn_dim} \
  --encoder-attention-heads ${num_head} \
  --decoder-embed-dim ${d_model} \
  --decoder-ffn-embed-dim ${fnn_dim} \
  --decoder-attention-heads ${num_head} \
  --encoder-layers 10 \
  --decoder-layers 2 \
  --activation-fn swish \
  --max-update 100000 \
  --source-lang en --target-lang de \
  --share-decoder-input-output-embed \
  --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
  --lr ${lr} --lr-scheduler inverse_sqrt --warmup-updates 16000 \
  --dropout 0.1 --attention-dropout 0.1 --drophead-prob 0.1 --weight-decay 0.0001 \
  --criterion label_smoothed_cross_entropy \
   --label-smoothing 0.1 \
  --max-tokens 4096 \
  --update-freq 4 \
  --seed 42 \
  --keep-interval-updates 4 \
  --validate-interval-updates 2000 \
  --keep-best-checkpoints 4 \
  --eval-bleu \
  --eval-bleu-detok moses \
  --eval-bleu-remove-bpe \
  --eval-bleu-print-samples \
  --best-checkpoint-metric bleu \
  --maximize-best-checkpoint-metric \
  --amp --ema-decay 0.9999 \
  --patience 10 \
  --save-dir checkpoints/ende-base-transformer \

