# PR-VIST
This repository is the implementation of PR-VIST:

Plot and Rework: Modeling Storylines for Visual Storytelling (ACL-IJCNLP2021 Findings) [[arXiv]](https://arxiv.org/abs/2105.06950)

### Download PR-VIST Generated Stories
> Working directory: `PRVIST/generated_stories/`
```bash=
unzip PR_VIST.json.zip
```
If you only want our generated stories:
`predicted_story` in the `PR_VIST.json` is the whole story predicted by PR-VIST. 
However, if you want to learn more about PR-VIST, please read the following:
## Environment
```
pytorch==1.7.1
python==3.7.6
```

## STAGE 0: Preparation
#### Download dataset and knowledge graphs
* GloVe Word Embedding
* Relation ids
* VIST-External_datasets: 
  * VIST and Visual Genome Knowledge Graph, and the combined VIST+VG (Multi) Knowledge Graph .
  * Extracted VIST images' objects
  * Extracted VIST images' terms
* VIST-Golden:
  * VIST golden story
  * VIST golden story path

> Working directory: `PRVIST/story_plotting/`
```bash=
bash download_big_data.sh
unzip data.zip
```
## STAGE 1: Story Plotting
#### STEP A. Train Storyline Predictor: 
> Working directory: `PRVIST/story_plotting/script`
```bash=
bash run_once.sh
```
trained model checkpoints will be saved as: `PRVIST/story_plotting/saved_model/`



#### STEP B. Generate Storylines:
If you wish to use our pre-trained storyline predictor model checkpoint instead, you can download it via: 

> Working directory: `PRVIST/story_plotting/`
```bash=
bash download_checkpoint.sh

cd saved_model
unzip HR_BiLSTM_plus_36.zip
```

Generate storylines:

> Working directory: `PRVIST/story_plotting/script`

Open the file: `run_generation.sh`, and edit the `--path ../saved_model/HR_BiLSTM_plus_432 ` to your desire model path. e.g.) change `--path ../saved_model/HR_BiLSTM_plus_432 ` to `--path  ../saved_model/HR_BiLSTM_plus_36`

Then, 
```bash=
bash run_generation.sh
```

The predicted storyline will be saved as: `../generated_storylines/pred_terms_[......].json`

#### (Optional) Download Predicted Storylines:
> Working directory: `PRVIST/story_plotting/`

Training Story-predictor may take a while, if you wish to skip STEP A-B, you can download our predicted storylines directly via:

```bash=
bash download_example.sh
unzip generated_storylines.zip
```

## STAGE 2: Story Reworking
 The implemented Transformer in this paper is: 
 Length-Controlled Transformer (proposed in  ACL-IJCNLP demo 2021: Stretch-VST: Getting Flexible With Visual Stories). 
 
 #### STEP A. Download Datasets
> Working directory: `PRVIST/story_reworking/`
```bash=
bash download_big_data.sh
unzip data.zip
```
 #### STEP B. Download Discriminator Model Checkpoints
> Working directory: `PRVIST/story_reworking/discriminator/`
```bash=
bash download_checkpoint.sh
unzip saved_model.zip
```

 #### STEP C. Pre-Train Transformer with ROC Story dataset 
> Working directory: `PRVIST/story_reworking/`

```bash=
bash run.sh [YOUR_DEVICE_NUMVER] roc
```
e.g., 
if you want to train on GPU device 0
```bash=
bash run.sh 0 roc
```

the trained model checkpoint is saved as: `save_model_BIO_[TODAY's DATE]/trained.chkpt`

#### STEP D. Finetune Transformer on VIST dataset
> Working directory: `PRVIST/story_reworking/`

```bash=
bash run_finetune.sh [MODEL_CHECKPOINT_FILEPATH] finetune [YOUR_DEVICE_NUMVER]
```
e.g., 
YOUR_DEVICE_NUMVER = 1, 
MODEL_CHECKPOINT_FILEPATH = save_model_BIO_August18roc1.5_reverse/trained.chkpt

```bash=
bash run_finetune.sh save_model_BIO_August18roc1.5_reverse/trained.chkpt finetune 1
```

the trained model checkpoint is saved as: `save_model_BIO_[TODAY’s DATE]_hierarchical_story_dis_vist/[xx.xxx].chkpt
`
where xx.xxx = validation perplexity

#### STEP E. Story Generation
> Working directory: `PRVIST/story_reworking/`
If you wish to use our pre-trained model checkpoint instead: 
```bash=
bash download_checkpoint.sh
unzip save_model_BIO_Jul-14-2021finetune1.5_hierarchical_story_dis_reward_rate_1.0_pretrain_vist.zip
```
This checkpoint was pre-trained on ROC and finetuned on VIST with discriminator.  

Generate stories from the predicted storylines:
```bash=
python 1sentence_inference.py -model [MODEL_CHECKPOINT_FILEPATH] -device [YOUR_DEVICE_NUMVER] -hop 1.5 -add_term True -term_path [Predicted_Storyline]
```

Example code:
```bash=
python 1sentence_inference.py -model save_model_BIO_Jul-14-2021finetune1.5_hierarchical_story_dis_reward_rate_1.0_pretrain_vist/trained.chkpt -device 2 -hop 1.5 -add_term True -term_path ../../story_plotting/generated_storylines/example_storyline.json
```

output filename = f'generated_story/TransLSTM{str(opt.hop)}_{model_path}_term_{term_path}.json'

---
## Side notes:
I would upload the rest of the model checkpoints in the future!

### Stage 1
* If your UHop training is very slow, it's perfectly normal!!! I took roughly a day to train an epoch. It's not very computationally efficient, but it's probably one of the fastest framework avaliable.
* I tried training with several different parameter settings (not all, because it's very computationally expensive), and it seems **unlikely to have any effect on model performance**.

