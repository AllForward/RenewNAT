# RenewNAT (AAAI 2023)
Some codes are borrowed from GLAT (https://github.com/FLC777/GLAT).

### Requirements

* Python >= 3.7
* Pytorch >= 1.5.0
* Fairseq 1.0.0a0

### Preparation
Train an autoregressive Transformer according to the instructions in [Fairseq](https://github.com/pytorch/fairseq).

Use the trained autoregressive Transformer to generate target sentences for the training set.

Binarize the distilled training data.

```
input_dir=path_to_raw_text_data
data_dir=path_to_binarized_output
src=source_language
tgt=target_language
python3 fairseq_cli/preprocess.py --source-lang ${src} --target-lang ${tgt} --trainpref ${input_dir}/train \
    --validpref ${input_dir}/valid --testpref ${input_dir}/test --destdir ${data_dir}/ \
    --workers 32 --src-dict ${input_dir}/dict.${src}.txt --tgt-dict {input_dir}/dict.${tgt}.txt
```

### Train
* For training RenewNAT w/ GLAT
```
data_dir=path_for_saving_dataset
save_path=path_for_saving_models
python3 train.py ${data_dir} --arch renewnat_glat --noise full_mask --share-all-embeddings \
    --criterion renewnat_loss --label-smoothing 0.1 --lr 5e-4 --warmup-init-lr 1e-7 --stop-min-lr 1e-9 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --optimizer adam --adam-betas '(0.9, 0.999)' \
    --adam-eps 1e-6 --task translation_renewnat --max-tokens 8192 --weight-decay 0.01 --dropout 0.1 \
    --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512 --fp16 \
    --max-source-positions 1000 --max-target-positions 1000 --max-update 300000 --seed 0 --clip-norm 5\
    --save-dir ${save_path} --src-embedding-copy --length-loss-factor 0.05 --log-interval 100 \
    --eval-bleu --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
    --eval-tokenized-bleu --eval-bleu-remove-bpe --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric --decoder-learned-pos --encoder-learned-pos \
    --apply-bert-init --activation-fn gelu --user-dir renewnat_plugins --mlm-layers 2
```

* For training RenewNAT w/ Vanilla NAT
```
data_dir=path_for_saving_dataset
save_path=path_for_saving_models
python3 train.py ${data_dir} \
    --arch renewnat_vanilla_nat --noise full_mask --share-all-embeddings \
    --criterion nat_loss --label-smoothing 0.1 --lr 5e-4 --warmup-init-lr 1e-7 --stop-min-lr 1e-9 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --optimizer adam --adam-betas '(0.9, 0.98)' \
    --adam-eps 1e-6 --task translation_renewnat --max-tokens 8192 --weight-decay 0.01 --dropout 0.1 \
    --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512 --fp16 \
    --max-source-positions 1000 --max-target-positions 1000 --max-update 300000 --seed 0 --clip-norm 5\
    --save-dir ${save_path} --src-embedding-copy --length-loss-factor 0.05 --log-interval 100 \
    --eval-bleu --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
    --eval-tokenized-bleu --eval-bleu-remove-bpe --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric --decoder-learned-pos --encoder-learned-pos \
    --apply-bert-init --activation-fn gelu --user-dir renewnat_plugins --mlm-layers 2
```

- For training RenewNAT w/ GLAT + DSLP

```
data_dir=path_for_saving_dataset
save_path=path_for_saving_models
src=src_language
tgt=tgt_language
python3 train.py ${data_dir} --source-lang ${src} --target-lang ${tgt}  --save-dir ${save_path} --eval-tokenized-bleu \
   --keep-interval-updates 5 --maximize-best-checkpoint-metric --eval-bleu-remove-bpe --eval-bleu-print-samples \
   --best-checkpoint-metric bleu --log-format simple --log-interval 100 --ddp-backend=no_c10d \
   --eval-bleu --eval-bleu-detok space --keep-last-epochs 5 --keep-best-checkpoints 5  --fixed-validation-seed 7 \
   --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
   --share-all-embeddings --decoder-learned-pos --encoder-learned-pos  --optimizer adam --adam-betas "(0.9,0.98)" --lr 0.0005 \ 
   --lr-scheduler inverse_sqrt --stop-min-lr 1e-09 --warmup-updates 10000 --warmup-init-lr 1e-07 \
   --apply-bert-init --weight-decay 0.01 --fp16 --clip-norm 2.0 --max-update 300000  --task translation_renewnat_dslp \ 
   --criterion renewnat_loss --arch renewnat_glat_dslp --noise full_mask \ 
   --concat-yhat --concat-dropout 0.0  --label-smoothing 0.1 \ 
   --activation-fn gelu --dropout 0.1  --max-tokens 8192 --glat-mode glat \ 
   --length-loss-factor 0.1 --pred-length-offset --user-dir renewnat_plugins --mlm-layers 2
```



### Inference

We average the 5 best checkpoints chosen by validation BLEU scores as our final model for inference. The script for averaging checkpoints is scripts/average_checkpoints.py

```
checkpoint_path=path_to_your_checkpoint
data_dir=path_to_your_dataset
src=src_language
tgt=tgt_language
python3 fairseq_cli/generate.py ${data_dir} --path ${checkpoint_path} --user-dir renewnat_plugins \
    --task translation_renewnat --remove-bpe --source-lang ${src} --target-lang ${tgt} --max-sentences 20   \
    --iter-decode-max-iter 0 --iter-decode-force-max-iter --iter-decode-eos-penalty 0 --iter-decode-with-beam 1 \
    --gen-subset test \
```

