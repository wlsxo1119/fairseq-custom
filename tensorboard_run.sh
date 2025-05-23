#model_path=checkpoints/ende-als-v2-transformer
model_path=checkpoints/ende-mto-transformer
#model_path=checkpoints/ende-base-transformer
#model_path=checkpoints/ende-label-smoothing-transformer
tensorboard_path=${model_path}/tensorboard
tensorboard --logdir=${tensorboard_path} --port 6006 --host 0.0.0.0
