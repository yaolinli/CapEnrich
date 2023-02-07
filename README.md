# CapEnrich

This is the official PyTorch implementation for the WWW 2023 [paper](https://arxiv.org/abs/2211.09371):

***CapEnrich: Enriching Caption Semantics for Web Images via Cross-modal Pre-trained Knowledge***



We provide the codes of our plug-and-play framework CapEnrich taking [VinVL (Oscar+)](https://arxiv.org/abs/2101.00529) as the Vision-Language-Pretraining(VLP) backbone. Our codes are built on the [VinVL repo](https://github.com/microsoft/Oscar).



## Requirements

First  install the requirements that VinVL needs referring to its [INSTALL.md](https://github.com/microsoft/Oscar/blob/master/INSTALL.md).

Then install other requirements and the [CLIP](https://github.com/openai/CLIP):

```
$ conda activate oscar
$ pip install ftfy regex tqdm spacy
$ pip install git+https://github.com/openai/CLIP.git
```

Install the coco_caption evaluation codes:

```
pip install git+https://github.com/jmhessel/pycocoevalcap.git
```



## Download

Download the image features,  text annotations of MSCOCO dataset and the released pre-trained model of VinVL available at its [repo page](https://github.com/microsoft/Oscar/blob/master/VinVL_DOWNLOAD.md).

The raw images,  region features, annotations of MSCOCO datasets  should be put in `./oscar/datasets/`

The official released VinVL_base  model (after CE and RL two-stage fine-tuning on MSCOCO dataset)  should be put in  `./oscar/pretrained_model/`



## Automatic Data-building

Construct new-format data like "generic caption, details"  on the MSCOCO dataset:

1) Extract scene graph of all annotations using the [tool](https://github.com/vacancy/SceneGraphParser):

```
# install scene graph parser tool 
pip install SceneGraphParser
python -m spacy download en
# get scene graphs
cd process_data/
python get_scenegraphs.py
```

2) Aggregate multiple annotations to a more detailed one based on the scene graphs

```
python newdata_construct.py
```



## Training with Learnable Prompts

Refer to `run.sh`  and the specific commands are as followings:

```
cd ..
python setup.py build develop
cd oscar

CUDA_VISIBLE_DEVICES=3 python run_captioning.py \
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
    --output_dir experiments/output_3e-4_nctx2_random
```

the number of prompts can be set by  `--n_ctx`  such as 2,4,6,8,  default is 2. 

the initialization of prompts can be set by  `--ctx_init`,  1) random initialization from a zero-mean
Gaussian distribution `--ctx_init ''` or 2) initialization from specified word embeddings, such as `--ctx_init 'the man'`



## Inference

Refer to `inference.sh` and set the checkpoint path  `--eval_model_dir`

```
# generate more details on test set
CUDA_VISIBLE_DEVICES=4 python end_uni_predict.py \
    --do_predict \
    --predict_yaml test.yaml \
    --per_gpu_eval_batch_size 1 \
    --num_beams 5 \
    --max_gen_length 40 \
    --data_dir ./datasets/coco_caption \
    --output_dir eval_results \
    --output_file output_3e-4_nctx2_random.json \
    --eval_model_dir experiments/output_3e-4_nctx2_random/best_checkpoint \
    --caption_file './eval_results/vinvl_result.json'

# aggregating multiple generated captions
cd process_data/
python post_process.py
```

The generated captions are available at  `./oscar/eval_results/`



## Evaluation

Run the *accuracy* captioning metrics including SPICE, CLIPScore and [Ref-CLIPScore](https://github.com/jmhessel/clipscore) as followings:

```
cd metrics/clipscore
python eval.py  --testfile your_test_file  --annofile your_gt_file
```

An example is:

```
python eval.py  --testfile '../../eval_results/vinvl_result.json'  --annofile  '../../datasets/coco_caption/test_caption_coco_format.json' 
```

We also provide the codes to calculate the refined CLIP R@K score on the *Hard Retrieval Pool*.

```
cd metrics/clip_Self_retrieve
python coco_process_t2i_sim.py --testfile ../../eval_results/vinvl_result.json  --retrieve_set hard
```



## Citation

```
@inproceedings{Yao2022CapEnrichEC,
  title={CapEnrich: Enriching Caption Semantics for Web Images via Cross-modal Pre-trained Knowledge},
  author={Linli Yao and Weijing Chen and Qin Jin},
  booktitle={{TheWebConf}},
  year = {2023}
}
```

