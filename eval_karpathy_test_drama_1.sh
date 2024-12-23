python eval_drama_1.py --dump_images 0  \
	--num_images 2544  \
        --infos_path /data/fjq/drama_log/log_transformer_drama_cap/infos_transformer.pkl \
	--model /data/fjq/drama_log/log_transformer_drama_cap/model.pth \
	--language_eval 1  \
	--batch_size 10   \
	--split test       \
	--max_length 120   \
	--temperature 0.1  \
	--top_p 0.75  \
	--drama_img_path  /data/fjq/DRAMA/imgs/   \
	--save_bbox_img_path  /data/fjq/drama_log/log_transformer_drama_cap/regs/   \
	--save_path_seq /data/fjq/drama_log/log_transformer_drama_cap/eval_karpathy_test_seq_drama.json \
	--save_path_index_iou /data/fjq/drama_log/log_transformer_drama_cap/eval_karpathy_test_index_iou_drama.json  \
