# Reading-Order-Relation-Enhanced Methods

## Preparation

Please follow the instructions from `make_weights/README.md` to prepare the pre-trained models, and configure their local paths in the shells. 

## Scripts for tasks

### Semantic Entity Recognition (SER) and Entity Linking (EL)

Please use the ROOR dataset, or automatically annotate other datasets (FUNSD, CORD, SROIE, etc.) with our ROP model.

* Running LayoutLMv3 for SER: `shells/{train,test}_v3_ner.sh`
* Running LayoutLMv3 for EL: `shells/{train,test}_v3_ner.sh`
* Running GeoLayoutLM for both tasks: `shells/{train,test}_geolayoutlm_ie.sh`

### DocVQA

Due to the policies of the fine-tuning datasets, we are unable to provide the processed fine-tuning datasets in this repository.
Please construct the fine-tuning data corresponding to sample data at `data/docvqa`, and then running `shells/{train,test}_lmv3_docvqa.sh` for fine-tuning and validation. 
