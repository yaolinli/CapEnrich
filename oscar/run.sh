###############################################
#####              VinVL                #######
###############################################
cd ..
python3.7 setup.py build develop
cd oscar


CUDA_VISIBLE_DEVICES=3 python3.7 run_captioning.py \
    --model_name_or_path ./pretrained_model/coco_captioning_base_scst/checkpoint-15-66405 \
    --do_train \
    --do_lower_case \
    --add_od_labels \
    --learning_rate 3e-4 \
    --per_gpu_train_batch_size 48 \
    --num_train_epochs 30 \
    --tie_weights \
    --freeze_embedding \
    --label_smoothing 0.1 \
    --drop_worst_ratio 0.2 \
    --drop_worst_after 20000 \
    --caption_file './datasets/{}_prefix_prompts.json' \
    --data_dir './datasets/coco_caption' \
    --evaluate_during_training \
    --save_epochs 1 \
    --n_ctx 2 \
    --ctx_init "" \
    --output_dir experiments/output_3e-4_nctx2

