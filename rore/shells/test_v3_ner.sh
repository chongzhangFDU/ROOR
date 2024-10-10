export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/path/to/ROOR/rore
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

CUDA_VISIBLE_DEVICES=0 python tasks/train_v3_ner.py \
  --do_train false \
  --do_test true \
  --image_dir /path/to/dataset \
  --json_dir /path/to/dataset/jsons \
  --split_file_dir /path/to/dataset \
  --test_dataset data.test.txt \
  --label_file_name labels.txt \
  --pretrained_model_path /path/to/dataset/layoutlmv3-base-2048 \
  --save_model_dir /path/to/save_model_dir \
  --lam 1e-2 \
  --num_ro_layers 12 \
  --use_aux_ro true \
  --transitive_expand false \
  --batch_size 1 \
  --shuffle false \
  --gpus 1