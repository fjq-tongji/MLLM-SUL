id="transformer_drama_reg"
if [ ! -f log/log_$id/infos_$id.pkl ]; then
start_from=""
else
start_from="--start_from log/log_$id"
fi
python train_drama_1-reg.py --id transformer  \
    --caption_model transformer \
    --refine 1 \
    --refine_aoa 1 \
    --use_ff 0 \
    --decoder_type AoA \
    --use_multi_head 2 \
    --num_heads 8 \
    --multi_head_scale 1 \
    --mean_feats 1 \
    --ctx_drop 1 \
    --dropout_aoa 0.3 \
    --label_smoothing 0.2 \
    --pretrained_llama /data/fjq/LLAMA/llama-2-7b/consolidated.00_1.pth  \
    --pretrained_reg  /data/fjq/drama_log/log_transformer_0304_swin-b/model-stage1.pth       \
    --input_json data/drama/drama_llama_adapter_3_r4_0211.json \
    --input_label_h5 data/drama/drama_llama_adapter_3_r4_0211.h5 \
    --h5_reg  True   \
    --input_att_dir  /data/fjq/DRAMA/att  \
    --input_grid_dir  /data/fjq/DRAMA/imgs_features_res101_224 \
    --input_global_dir  /data/fjq/DRAMA/imgs_features_swin_large_384_2 \
    --seq_per_img 1  \
    --batch_size 64  \
    --beam_size 2  \
    --use_box   0  \
    --learning_rate 1e-5 \
    --num_layers 6 \
    --input_encoding_size 512 \
    --rnn_size 2048 \
    --learning_rate_decay_start 0 \
    --learning_rate_decay_rate 0.8 \
    --scheduled_sampling_start 0 \
    --checkpoint_path /data/fjq/drama_log/log_$id  \
    $start_from   \
    --save_checkpoint_every 3000  \
    --language_eval 1 \
    --val_images_use -1 \
    --max_epochs 150 \
    --save_history_ckpt 0   \
    --scheduled_sampling_increase_every 5 \
    --scheduled_sampling_max_prob 0.5 \
    --learning_rate_decay_every -1