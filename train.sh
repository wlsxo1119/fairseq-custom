d_model=512
fnn_dim=$(echo "4 * ${d_model}" | bc -l)
num_head=8
lr=0.0004
criterion=margin_token_loss
update_step=100000
save_dir=checkpoints/ende-mto-als-transformer
max_token=2048
update_freq=8
echo "d_model:" ${d_model}
echo "fnn_dim:" ${fnn_dim}
echo "num_head:" ${num_head}
echo "lr:" ${lr}
echo "update_step:" ${update_step}
echo "save_dir:" ${save_dir}
echo "max_token:" ${max_token}
echo "update_freq:" ${update_freq}
echo "criterion:" ${criterion}

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
  --source-lang en --target-lang de \
  --share-decoder-input-output-embed \
  --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
  --lr ${lr} --lr-scheduler inverse_sqrt --warmup-updates 16000 \
  --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.0001 \
  --criterion ${criterion} \
  --max-update ${update_step}  \
  --max-tokens ${max_token} \
  --update-freq ${update_freq} \
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
  --ema-decay 0.9999 \
  --patience 10 \
  --tensorboard-logdir ${save_dir}/tensorboard \
  --save-dir ${save_dir}
