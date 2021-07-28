# PR-VIST
This repository is the implementation of PR-VIST:

Plot and Rework: Modeling Storylines for Visual Storytelling (ACL-IJCNLP2021 Findings) [[arXiv]](https://arxiv.org/abs/2105.06950)

### Generated Stories
> Working directory: `PRVIST/generated_stories/`
```bash=
unzip PR_VIST.json.zip
```
`predicted_story` is the whole story predicted by PR-VIST

## Environment
```
pytorch==1.7.1
python==3.7.6
```

## Stage 0: Preparation
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
## Stage 1: Story Plotting
#### A. Training Storyline Predictor: 
> Working directory: `PRVIST/story_plotting/script`
```bash=
bash run_once.sh
```
generated model checkpoints will be saved to: `PRVIST/story_plotting/saved_model/`



#### B. Generating Storyline:
For pre-trained storyline predictor: 

> Working directory: `PRVIST/story_plotting/`
```bash=
bash download_checkpoint.sh

cd saved_model
unzip HR_BiLSTM_plus_36.zip
```

Generating storyline:

> Working directory: `PRVIST/story_plotting/script`

Open the file: `run_generation.sh`

Edit the `--path ../saved_model/HR_BiLSTM_plus_432 ` to your desire model path. 

e.g.) change `--path ../saved_model/HR_BiLSTM_plus_432 ` to `--path  ../saved_model/HR_BiLSTM_plus_36`

Then, 
```bash=
bash run_generation.sh
```

The predicted storyline will be saved to: `../generated_storylines/pred_terms_[......].json`

#### C. Download Predicted Storyline:
> Working directory: `PRVIST/story_plotting/`

Download the predicted storyline in this paper:

```bash=
bash download_example.sh
unzip generated_storylines.zip
```
## Stage 2: Story Reworking
 The implemented Transformer in this paper is: 
 Length-Controlled Transformer (proposed in  ACL-IJCNLP demo 2021: Stretch-VST: Getting Flexible With Visual Stories). 
 
 #### A. Download Datasets
> Working directory: `PRVIST/story_reworking/`
```bash=
bash download_big_data.sh
unzip data.zip
```
 #### B. Download Discriminator Model Checkpoints
> Working directory: `PRVIST/story_reworking/discriminator/`
```bash=
bash download_checkpoint.sh
unzip data.zip
```

 #### C. Pre-Train Transformer with ROC Story dataset 
> Working directory: `PRVIST/story_reworking/`

```bash=
bash run.sh [YOUR_DEVICE_NUMVER] roc
```
e.g., 
if you want to train on GPU device 0
```bash=
bash run.sh 0 roc
```

the trained model checkpoint is saved to: `save_model_BIO_[TODAY's DATE]/trained.chkpt`

#### D. Finetuning on VIST dataset
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

the trained model checkpoint is saved to: `save_model_BIO_[TODAY’s DATE]_hierarchical_story_dis_vist/[xx.xxx].chkpt
`
where xx.xxx = validation perplexity

#### E. Story Generation (For pre-trained Transformer checkpoints: coming soon...!)
> Working directory: `PRVIST/story_reworking/`

```bash=
python 1sentence_inference.py -model [MODEL_CHECKPOINT_FILEPATH] -device [YOUR_DEVICE_NUMVER] -hop 1.5 -add_term True -term_path [Predicted_Storyline]
```

Example code:
```bash=
python 1sentence_inference.py -model save_model_BIO_August18finetune1.5_hierarchical_reverse_story_dis_sen_dis_pretrain_vist/trained_ppl_61.621.chkpt -device 2 -hop 1.5 -add_term True -term_path ../../story_plotting/generated_storylines/example_storyline.json
```

output filename = f'generated_story/TransLSTM{str(opt.hop)}_{model_path}_term_{term_path}.json'

---
## Side notes:
I would upload the rest of the model checkpoints in the future!

### Stage 1
* If your UHop training is very slow, it's perfectly normal!!! I took roughly a day to train an epoch. It's not very computationally efficient, but it's probably one of the fastest framework avaliable.
* I tried training with several different parameter settings (not all, because it's very computationally expensive), and it seems **unlikely to have any effect on model performance**.

