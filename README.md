# PR-VIST
This repository is the implementation of PR-VIST
**Plot and Rework: Modeling Storylines for Visual Storytelling**

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

Download data via:  
```bash=
cd story_plotting
bash download_big_data.sh
unzip data.zip
```
## Stage 1: Story Plotting
