# wp-hungarian
The code in this repository is a support for the experiments in the paper [On a Novel Application of Wasserstein-Procrustes for Unsupervised Cross-Lingual Learning](https://arxiv.org/abs/2007.09456)
RUNNING DIRECTIONS (GPU is required)
Code “iterative_hungarian.py” takes one initialisation matrix W_0 and refines it. 

Experiments from section 5.1 are recreated the following way (this example shows English-Spanish):
1. Obtaining the initialisation matrix
MUSE (https://github.com/facebookresearch/MUSE): `python unsupervised.py --src_lang en --tgt_lang es --src_emb data/wiki.en.vec --tgt_emb data/wiki.es.vec --n_refinement 5`

Procrustes (https://github.com/facebookresearch/MUSE): `python supervised.py --src_lang en --tgt_lang es --src_emb data/wiki.en.vec --tgt_emb data/wiki.es.vec --n_refinement 5 --dico_train default`

ICP (https://github.com/facebookresearch/Non-adversarialTranslation): `python get_data.py
python run_icp.py
python eval.py`

2. Running IH 
The source and target embeddings can be downloaded in the following way (change link for other languages):
# English fastText Wikipedia embeddings
`curl -Lo wiki.en.vec https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec`
# Spanish fastText Wikipedia embeddings
`curl -Lo wiki.es.vec https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.es.vec`

Running IH:
`python iterative_hungarian.py —-grows 45000 —-write_path AUX —-src_path PATH_SRC_EMBEDDINGS —-tgt_path PATH_TGT_EMBEDDINGS —-w_path PATH_INITIALIZATION_MATRIX` 

3. Final refinements and evaluation: 

These are done via MUSE (https://github.com/facebookresearch/MUSE):
`python unsupervised.py --src_lang en --tgt_lang es  --src_emb wiki.en.vec --tgt_emb wiki.es.vec --n_refinement 5 —-adversarial False --exp_path PATH —-exp_name TO —-exp_id EXPERIMENT`
We are assuming the transformation matrix is saved in PATH/TO/EXPERIMENT/best_mapping.pth (MUSE needs non-empty values for —-exp_path, —-exp_name and —-exp_id)

Experiments from section 5.2 are recreated the following way:
1) Word embeddings are obtained using Fasttext following instructions in https://arxiv.org/abs/1805.11222 
2) `python iterative_hungarian.py —-grows 10000 —-write_path AUX —-src_path PATH_SRC_EMBEDDINGS —-tgt_path PATH_TGT_EMBEDDINGS —-w_path PATH_INITIALIZATION_MATRIX`
