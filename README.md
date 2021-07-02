# PR-VIST
This repository is the implementation of PR-VIST
**Plot and Rework: Modeling Storylines for Visual Storytelling**

## Environment
```
pytorch==1.7.1
python==3.7.6
```

## Stage 0: Preparation
### Download dataset and knowledge graphs
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
### A. Training Storyline Predictor: 
> Working directory: `PRVIST/story_plotting/script`
```bash=
bash run_once.sh
```
generated model checkpoints will be saved to: `PRVIST/story_plotting/saved_model/`

For pre-trained checkpotins: coming soon...!

### B. Generating storyline:
**Model Path Customization:**
> Working directory: `PRVIST/story_plotting/script`
Open the file: `run_generation.sh`
Edit the `--path ../saved_model/HR_BiLSTM_plus_432 ` to your desire model path in saved_model/. 
e.g.) change `--path ../saved_model/HR_BiLSTM_plus_432 ` to `--path  ../saved_model/HR_BiLSTM_plus_1`

**Run UHop generation:**
> Working directory: `PRVIST/story_plotting/script`
```bash=
bash run_generation.sh
```

The predicted storyline is saved to: `../generated_storylines/pred_terms_[......].json`

**Download our storyline via:**
> Working directory: `PRVIST/story_plotting/`
```bash=
bash download_example.sh
unzip generated_storylines.zip
```
