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

Download them via:  
```bash=
cd story_plotting
bash download_big_data.sh
unzip data.zip
```
## Stage 1: Story Plotting
