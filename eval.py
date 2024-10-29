import pandas as pd
import numpy as np
import os
from compare_all import dat_names
from utils.all_gene import ALL_INTER_GENE 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm
import xgboost as xgb
from scipy.stats import loguniform
from pycaret.classification import *
from pycaret.classification import ClassificationExperiment

RES_SAVE_PATH = "result"
BASE_SAVE_PATH = "dataset"
dat_names = ['genept_imp_0']
task_names =  ['loc']
ML_MODEL = 'logistic'

EVAL_ML = True
EVAL_CCA = True


def main():
    for task_name in task_names:
        for dat_name in dat_names:
            dat = np.load(os.path.join(BASE_SAVE_PATH, 'biolinkbert_genename.npy'))
            dat_gene = np.load(os.path.join(BASE_SAVE_PATH, 'biolinkbert_genename_gene.npy'))
            task_eval_dict[task_name](dat, dat_gene, dat_name)
            print(dat_name, task_name)        


def eval_ppi(dat, dat_gene, dat_name):
    # load data
    ppi_dat = pd.read_csv(os.path.join(BASE_SAVE_PATH, 'ppi.tsv'), sep='\t')
    # remove duplicate genes
    ppi_dat = ppi_dat.rename(columns={'gene': 'gene_symbol'})

    # intersect
    ppi_dat_gene = list(set(ppi_dat['protein1']).union(set(ppi_dat['protein2'])))
    ppi_dat_gene = [gene for gene in ppi_dat_gene if gene in ALL_INTER_GENE]

    inter_gene = np.intersect1d(ppi_dat_gene, dat_gene)
    # print('number of intersected genes ', len(inter_gene))
    dat = dat[np.array([i for i,gene in enumerate(dat_gene) if gene in inter_gene]), :]
    dat_gene = [gene for gene in dat_gene if gene in inter_gene]

    # filter ppi gene
    ppi_dat = ppi_dat[ppi_dat['protein1'].isin(inter_gene)]
    ppi_dat = ppi_dat[ppi_dat['protein2'].isin(inter_gene)]

    ys = ppi_dat['combined_score'].values
    # construct Xs
    gene_to_index = {gene: idx for idx, gene in enumerate(dat_gene)}
    # Optimization 2: Preallocate the array
    Xs = np.empty((len(ppi_dat), dat.shape[1]))
    # Optimization 3: Vectorized operations
    for i, row in tqdm(enumerate(ppi_dat.itertuples(index=False)), total=len(ppi_dat)):
        idx1 = gene_to_index[row.protein1]
        idx2 = gene_to_index[row.protein2]
        Xs[i] = dat[idx1] * dat[idx2]

    save_name = 'PPI_'+dat_name
    pycaret_eval(Xs, ys, save_name)

    # auprc, shuffle_auprc = cross_validation(Xs, ys)
    # return (auprc, shuffle_auprc)


def eval_go(dat, dat_gene, dat_name):
    # load data
    go_dat = pd.read_csv(os.path.join(BASE_SAVE_PATH, 'go_BP.tsv'), sep='\t')
    # remove duplicate genes
    go_dat = go_dat.rename(columns={'gene': 'gene_symbol'})

    go_dat, dat, dat_gene = preprocess(go_dat, dat, dat_gene)
    go_scores = {}
    # evaluate disease separately
    # print('evaluate separately')
    for go in tqdm(go_dat.columns[1:]):
        go_dat_sub = go_dat[['gene_symbol',go]]
        go_dat_sub = go_dat_sub.iloc[list(np.argsort(go_dat_sub['gene_symbol']).values)]

        # assign fold
        ys = go_dat_sub[go].values
        Xs = dat
        save_name = 'go_'+ go+'_'+dat_name
        pycaret_eval(Xs, ys, save_name)

    return go_scores


def eval_loc(dat, dat_gene, dat_name):
    # load data
    loc = pd.read_csv(os.path.join(BASE_SAVE_PATH, 'deeploc.tsv'), sep='\t')

    # remove duplicate genes
    loc_list = loc.columns[:-1]
    loc_dat_gene = list(set(loc['gene']))
    loc_dat_gene = [gene for gene in loc_dat_gene if gene in ALL_INTER_GENE]
    inter_gene = np.intersect1d(loc_dat_gene, dat_gene)

    # print('number of intersected genes ', len(inter_gene))
    dat = dat[np.array([i for i,gene in enumerate(dat_gene) if gene in inter_gene]), :]
    dat_gene = [gene for gene in dat_gene if gene in inter_gene]
    loc = loc[loc['gene'].isin(inter_gene)]
    # sort
    dat = dat[np.argsort(dat_gene), :]
    loc = loc.iloc[np.argsort(loc['gene'].values.tolist()), :]

    # evaluate loc separately
    for lo in tqdm(loc_list):
        ys = loc[lo].values
        if sum(ys==1)==0 or sum(ys==0)==0:
            continue
        Xs = dat

        if lo=='Lysosome/Vacuole':
            lo='Lysosome|Vacuole'

        save_name = 'loc_'+ lo+'_'+dat_name
        pycaret_eval(Xs, ys, save_name)
        # auprc, shuffle_auprc = cross_validation(Xs, ys)
        # cca_score = cca_eval(Xs, ys)
        # go_scores[go] = (auprc, shuffle_auprc)


def eval_dis_gene(dat, dat_gene, dat_name):
    # load data
    dis_gene = pd.read_csv(os.path.join(BASE_SAVE_PATH, 'dis_gene_final.txt'), sep='\t')

    # remove duplicate genes
    dis_gene = dis_gene.drop_duplicates(subset=['gene_symbol', 'disease'])
    dis_gene, dat, dat_gene = preprocess(dis_gene, dat, dat_gene)
    dis_scores = {}

    # evaluate disease separately
    print('evaluate disease separately')
    for dis in tqdm(dis_gene['disease'].unique()):
        dis_gene_sub = dis_gene[dis_gene['disease']==dis]
        dis_gene_sub = dis_gene_sub.iloc[list(np.argsort(dis_gene_sub['gene_symbol']).values)]
        # assign fold
        ys = dis_gene_sub['target'].values
        Xs = dat
        save_name = 'dis_gene_'+ dis+'_'+dat_name
        pycaret_eval(Xs, ys, save_name)

        # auprc = cross_validation(Xs, ys)
        # dis_scores[dis] = auprc
    return dis_scores


def eval_dosage(dat, dat_gene, dat_name):
    # load data
    sensitive = pd.read_csv(os.path.join(BASE_SAVE_PATH, 'dosage_sensitive.txt'), header=None)
    insensitive = pd.read_csv(os.path.join(BASE_SAVE_PATH, 'dosage_insensitive.txt'), header=None)
    sensitive = list(sensitive[0])
    insensitive = list(insensitive[0])
    sensitive = [gene for gene in sensitive if gene in ALL_INTER_GENE]
    insensitive = [gene for gene in insensitive if gene in ALL_INTER_GENE]
    sen_dat = dat[np.array([i for i,gene in enumerate(dat_gene) if gene in sensitive]), :]
    insen_dat = dat[np.array([i for i,gene in enumerate(dat_gene) if gene in insensitive]), :]

    Xs = np.concatenate([sen_dat, insen_dat], axis=0)
    ys = np.concatenate([np.ones(len(sen_dat)), np.zeros(len(insen_dat))], axis=0)


    save_name = 'dosage_'+dat_name
    pycaret_eval(Xs, ys, save_name)

    # auprc = cross_validation(Xs, ys)
    # dis_scores[dis] = auprc


def eval_ggipnn(dat, dat_gene, dat_name):
    ggi = pd.read_csv(os.path.join(BASE_SAVE_PATH, 'ggipnn_gene_pairs.txt'), sep='\t', header=None)
    label = pd.read_csv(os.path.join(BASE_SAVE_PATH, 'ggipnn_labels.txt'), sep='\t', header=None)
    
    ggi_dat_gene = list(set(ggi[0]).union(set(ggi[1])))
    ggi_dat_gene = [gene for gene in ggi_dat_gene if gene in ALL_INTER_GENE]

    inter_gene = np.intersect1d(ggi_dat_gene, dat_gene)
    print('number of intersected genes ', len(inter_gene))
    dat = dat[np.array([i for i,gene in enumerate(dat_gene) if gene in inter_gene]), :]
    dat_gene = [gene for gene in dat_gene if gene in inter_gene]

    # filter ppi gene
    ggi['y'] = label[0].values
    ggi = ggi[ggi[0].isin(inter_gene)]
    ggi = ggi[ggi[1].isin(inter_gene)]
    ys = ggi['y'].values
    
    # construct Xs
    gene_to_index = {gene: idx for idx, gene in enumerate(dat_gene)}
    Xs = np.empty((len(ggi), dat.shape[1]))
    # Optimization 3: Vectorized operations
    for i, row in tqdm(enumerate(ggi.itertuples(index=False)), total=len(ggi)):
        idx1 = gene_to_index[row._0]
        idx2 = gene_to_index[row._1]
        Xs[i] = dat[idx1] * dat[idx2]

    save_name = 'ggipnn_'+dat_name
    pycaret_eval(Xs, ys, save_name)


def pycaret_eval(Xs, ys, save_name):
    n_dim = Xs.shape[1]
    dat = np.concatenate([Xs, ys.reshape(-1,1)], axis=1)
    columns = ['dim'+str(i) for i in range(n_dim)] + ['target']
    dat = pd.DataFrame(dat, columns=columns)
    s = setup(dat, target="target",  session_id = 123, use_gpu = True)
    add_metric('auprc', 'auprc', average_precision_score)

    exp = ClassificationExperiment()
    exp.setup(dat, target = 'target', session_id = 123)
    best = compare_models(sort='auprc', n_select=1, include = ['lr'])#, 'knn', 'nb', 'rf', 'mlp'])
    df = pull()
    df.to_csv(os.path.join(RES_SAVE_PATH, save_name+'_pycaret_results.tsv'), sep='\t')
    save_model(best, save_name)
    

def preprocess(eval_dat, dat, dat_gene): 
    # remove duplicate genes
    eval_dat = eval_dat[eval_dat['gene_symbol'].isin(ALL_INTER_GENE)]
    dat = dat[np.array([i for i,gene in enumerate(dat_gene) if gene in ALL_INTER_GENE]), :]
    dat_gene = [gene for gene in dat_gene if gene in ALL_INTER_GENE]

    # keep interected genes
    eval_dat_gene_list = eval_dat['gene_symbol'].unique()
    inter_gene = np.intersect1d(eval_dat_gene_list, dat_gene)
    print('number of intersected genes ', len(inter_gene))
    
    eval_dat = eval_dat[eval_dat['gene_symbol'].isin(inter_gene)]
    dat = dat[np.array([i for i,gene in enumerate(dat_gene) if gene in inter_gene]), :]
    dat_gene = [gene for gene in dat_gene if gene in inter_gene]

    # sort genes
    dat = dat[np.argsort(dat_gene), :]
    return eval_dat, dat, dat_gene
    

def cross_validation(Xs, ys, n_fold=5):
    # assign fold
    idx = np.arange(len(ys))
    np.random.shuffle(idx)
    fold = idx%n_fold

    auprcs = []
    shuffle_auprcs = []
    # five fold cross validation
    for i in range(n_fold):
        idx_train = fold!=i
        idx_test = fold==i
        X_train = Xs[idx_train, :]
        X_test = Xs[idx_test, :]
        y_train = ys[idx_train]
        y_test = ys[idx_test]
        
        # convert to binary
        y_train = np.array(y_train==True, dtype=int)
        y_test = np.array(y_test==True, dtype=int)
        
        accuracy, auprc = do_ml(X_train, X_test, y_train, y_test, ml_model=ML_MODEL)

        np.random.shuffle(y_train)
        accuracy, shuffle_auprc = do_ml(X_train, X_test, y_train, y_test, ml_model=ML_MODEL)
        auprcs.append(auprc)
        shuffle_auprcs.append(shuffle_auprc)
    return np.mean(auprcs), np.mean(shuffle_auprcs)


def do_ml(X_train, X_test, y_train, y_test, ml_model='xgboost'):
    if ml_model=='xgboost':
        param_dist = {
            "reg_lambda": loguniform(1e-2, 1e5), 
            "reg_alpha": loguniform(1e-2, 1e5)
        }
        reg = xgb.XGBClassifier(tree_method="hist")
        random_search = RandomizedSearchCV(
            reg, param_distributions=param_dist, n_iter=50, refit = True
        )
        
        mod = random_search.fit(
                X_train, y_train
            )

        y_pred = mod.predict_proba(
                X_test
            )
        
        accuracy, auprc = evaluate(y_pred, y_test)
    
    if ml_model=='logistic':
        clf = LogisticRegression(random_state=0).fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)[:,1]
        accuracy, auprc = evaluate(y_pred, y_test)
    
    
    return accuracy, auprc


def evaluate(y_pred, y_test):
    # print('accuracy', np.mean(y_pred==y_test))
    accuracy = np.mean(y_pred==y_test)

    # confusion matrix
    # print('confusion matrix' )
    # print(confusion_matrix(y_test, y_pred>0.5))

    # AUPRC
    # print('AUPRC')
    auprc = average_precision_score(y_test, y_pred)
    # print(auprc)

    return accuracy, auprc


task_eval_dict = {'dis_gene': eval_dis_gene, 'GO': eval_go, 
                  'dosage':eval_dosage,'ggipnn':eval_ggipnn,'loc':eval_loc,
                  'ppi': eval_ppi}


if __name__ == "__main__":
    main()
