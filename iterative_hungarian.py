import numpy as np
import torch
import pandas as pd
import scipy
from scipy import spatial
import csv
import sys
import ctypes
import subprocess 
from tqdm import tqdm
from multiprocessing import Pool
from lapsolver import solve_dense
import argparse
from .dico_builder import build_dictionary
csv.field_size_limit(sys.maxsize)

#creates the weight matrix
def make_matrix(src_vals, tgt_vals):
    src=torch.from_numpy(src_vals).cuda().float()
    tgt=torch.from_numpy(tgt_vals.transpose()).cuda().float()
    weight=src.mm(tgt)
    weight=weight.cpu().numpy()
    maxim=weight.max()
    return maxim-weight

#returns W that solves Procrustes
def procrustes(src_vals, tgt_vals):
    M=np.matmul(src_vals.transpose(), tgt_vals)
    U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
    W_np=np.matmul(U, V_t)
    return W_np

#solves the Hungarian
def solve(weights):
	rids, cids = solve_dense(weights)
	return cids
parser = argparse.ArgumentParser(description='input')
parser.add_argument('--src_path', type=str, help='path for source embeddings')
parser.add_argument('--tgt_path', type=str, help='path for target embeddings')
parser.add_argument('--nrows_init', type=int, help='number of word embeddings considered')
parser.add_argument('--nembed', type=int, help='dimension of the word embeddings', default=300)
parser.add_argument('--w_path', type=str, help='path for initial transformation matrix')
parser.add_argument('--write_path', type=str, help='path for writing results')
parser.add_argument('--nrefin', type=int, help='number of refinement steps', default=0)
args = parser.parse_args()

parser_dico = argparse.ArgumentParser(description='parameters of the dictionary')
parser_dico.add_argument('--dico_max_rank', type=int, default=15000)
parser_dico.add_argument('--dico_method', type=str, default='csls_knn_10')
parser_dico.add_argument('--dico_max_size', type=int, default=0)
parser_dico.add_argument('--dico_min_size', type=int, default=0)
parser_dico.add_argument('--dico_threshold', type=int, default=0)
parser_dico.add_argument('--dico_build', type=str, default='S2T')
parser_dico.add_argument('--cuda', type=bool, default=True)
params_dico = parser_dico.parse_args()

df_src = pd.read_csv(args.src_path, sep=' ', nrows=args.nrows_init, engine='python', skiprows=[0], header=None, quoting=csv.QUOTE_NONE, encoding = 'utf-8')
df_tgt = pd.read_csv(args.tgt_path, sep=' ',nrows=args.nrows_init, engine='python', skiprows=[0], header=None, quoting=csv.QUOTE_NONE, encoding = 'utf-8') 
src_words=df_src.iloc[:, 0].astype(str).values
tgt_words=df_tgt.iloc[:, 0].astype(str).values
src_vals=df_src.iloc[:, 1:args.nembed+1].values
tgt_vals=df_tgt.iloc[:, 1:args.nembed+1].values
nwords = min(src_vals.shape[0], tgt_vals.shape[0])
src_vals = src_vals/np.linalg.norm(src_vals, axis=1)[:, None]
tgt_vals = tgt_vals/np.linalg.norm(tgt_vals, axis=1)[:, None]
ITE=0
X=src_vals
Y=tgt_vals
maxim=0
rids=np.linspace(0, nwords-1, nwords).astype(int)
combo=[[' X,Y',' X, Yr'], [' Xr, Y',' Xr, Yr']]
W=torch.load(args.w_path)
X=np.matmul(X, W.transpose())
new_W=W.transpose()
REG=10000000 #use to deactivate the alternative problems
contin=True
while contin:
    contin=False
    print('_______________________')
    print('Iteration number '+str(ITE))
    X_red=X
    Y_red=Y
    if ITE<REG:
        U, S, V_t = scipy.linalg.svd(X, full_matrices=False)
        X_red=np.matmul(U, np.diag(S))
        U, S, V_t = scipy.linalg.svd(Y, full_matrices=False)
        Y_red=np.matmul(U, np.diag(S))
        weights=[make_matrix(X,Y), make_matrix(X,Y_red), make_matrix(X_red,Y), make_matrix(X_red,Y_red)]
        weights_neg = [x * (-1) for x in weights]
        weights=weights+weights_neg
        new_cids=[]
        print('Doing Hungarian')
        with Pool(32) as p:
            new_cids.append(p.map(solve, weights))
        print('Hungarian finished!')
    else: 
    	weights_new=make_matrix(X, Y)
    for idx1, x in enumerate([X, X_red]):
        for idx2, y in enumerate([Y, Y_red]):
            for sign in [1, -1]:
                passa=False
                if ITE<REG:
                    passa=True
                    cids=new_cids[0][idx1*2+idx2+2*(1-sign)]
                elif idx1==0 and idx2==0 and sign==1:
                    passa=True
                    cids=solve(weights_new)
                assert len(cids)==len(tgt_words)
                Y_aux=y[cids, :]				
                aux_norm=np.linalg.norm(np.array(np.matmul(x.transpose(), Y_aux)), ord='nuc')
                if(int(aux_norm)>int(maxim) and passa):
                    Xnew=x
                    print('Update norm: '+str(aux_norm))
                    print('Updated '+str(sign)+str(combo[idx1][idx2]))
                    Ynew=Y_aux
                    W=procrustes(x, Y_aux)
                    Xnew=np.matmul(Xnew, W)
                    maxim=aux_norm
                    new_W_aux=np.matmul(new_W, W)
                    np.save(args.write_path+'/X', Xnew)
                    np.save(args.write_path+'/Y', Ynew)
                    torch.save(new_W_aux.transpose(), args.write_path+'/best_mapping.pth')
                    new_tgt_words=tgt_words[cids]
                    np.save(args.write_path+'/tgt_words', new_tgt_words)					
                    if ITE!=0:
                        contin=(nwords-sum(cids==rids))
                        print('Distance wrt previous perm:'+str(args.nrows_init - sum(cids==rids)))
                    else: contin=True
    ITE=ITE+1
    if contin: 
        X=Xnew
        Y=Ynew
        tgt_words=new_tgt_words
        new_W=new_W_aux


if args.nrefin > 0:
    #we do the refinements
    for n_iter in range(args.nrefin):
        dico=build_dictionary(X, Y, params_dico) 
        w=procrustes(X[dico[:, 0]], Y[dico[:, 1]])
        torch.save(w.transpose(), args.write_path+'/best_mapping.pth')

    
