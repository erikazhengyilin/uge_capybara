import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import utils.cca_core as cca_core

# edit path here
RES_SAVE_PATH = "result"
BASE_SAVE_PATH = "dataset"
dat_names = ['biolinkbert_summary']

fit_svcca = True
PERMUTE = True
n_sample = 100

def process_dat(dat_i, dat_j, dat_names):
    dat1_name = dat_names[dat_i]
    dat2_name = dat_names[dat_j]
    dat1, dat1_gene = read_dat(dat1_name)
    dat2, dat2_gene = read_dat(dat2_name)

    gene_inter = set(dat1_gene).intersection(set(dat2_gene))

    # select the common genes
    dat1 = dat1[np.array([i for i,gene in enumerate(dat1_gene) if gene in gene_inter]), :]
    dat2 = dat2[np.array([i for i,gene in enumerate(dat2_gene) if gene in gene_inter]), :]

    # sort the data based on genes
    dat1_gene = [gene for gene in dat1_gene if gene in gene_inter]
    dat2_gene = [gene for gene in dat2_gene if gene in gene_inter]

    dat1 = dat1[np.argsort(dat1_gene), :]
    dat2 = dat2[np.argsort(dat2_gene), :]
    # dat1_gene = dat1_gene[np.argsort(dat1_gene)]
    # dat2_gene = dat2_gene[np.argsort(dat2_gene)]

    # standardize the data
    dat1 = (dat1 - np.mean(dat1, axis=0)) / np.std(dat1, axis=0)
    dat2 = (dat2 - np.mean(dat2, axis=0)) / np.std(dat2, axis=0)
    
    return dat1, dat2, dat1_name, dat2_name


def read_dat(data_type, n_dim=512):
    # example, for other files, follow the exact same format
    if data_type == 'biolinkbert_summary':
        if os.path.exists(os.path.join(BASE_SAVE_PATH, 'biolinkbert_genename.npy')):
            genename_fp = np.load(os.path.join(BASE_SAVE_PATH, 'biolinkbert_genename.npy'))
            gene_fp = np.load(os.path.join(BASE_SAVE_PATH, 'biolinkbert_genename_gene.npy'))
            return genename_fp, gene_fp
        else:
            raise FileNotFoundError("One or both required files (biolinkbert_genename.npy, biolinkbert_genename_gene.npy) do not exist in the specified path.")

        
def plot_cor_matrix(cor_matrix, dat_names, fig_name):
    # add transpose
    cor_matrix = cor_matrix + cor_matrix.T
    # set diagonal to 1
    np.fill_diagonal(cor_matrix, 1)
    
    plt.imshow(cor_matrix, cmap='viridis', interpolation='nearest')
    # put text in each cell
    for i in range(len(dat_names)):
        for j in range(len(dat_names)):
            plt.text(j, i, '{:.2f}'.format(cor_matrix[i, j]), ha='center', va='center', color='w')
    plt.xticks(np.arange(len(dat_names)), dat_names, rotation=90)
    plt.yticks(np.arange(len(dat_names)), dat_names)
    plt.tight_layout()

    plt.colorbar()
    plt.savefig(f'{RES_SAVE_PATH}/{fig_name}_cor_matrix.png', bbox_inches='tight')
    plt.show()
    plt.close()

def fit_svcca_model(dat1, dat2, permute=False):
    dat1 = dat1.T
    dat2 = dat2.T

    svcca_cor = cal_svcca(dat1, dat2)

    if permute:
        correlation_list = []

        for i in tqdm(range(n_sample)):
            dat1_perm = dat1[:, np.random.permutation(dat1.shape[1])]
            dat2_perm = dat2[:, np.random.permutation(dat2.shape[1])]
            correlation_list.append(cal_svcca(dat1_perm, dat2_perm))
        p_value = np.sum(np.array(correlation_list) > svcca_cor) / n_sample
        adj_svcca_cor = svcca_cor - np.mean(correlation_list)
    else:
        adj_svcca_cor = None
        p_value = None

    return svcca_cor, adj_svcca_cor, p_value


def cal_svcca(dat1, dat2):
    dat1 = dat1.astype(np.float64)
    dat2 = dat2.astype(np.float64)
    dat1 = dat1 - np.mean(dat1, axis=1, keepdims=True)
    dat2 = dat2 - np.mean(dat2, axis=1, keepdims=True)

    U1, s1, V1 = np.linalg.svd(dat1, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(dat2, full_matrices=False)

    # keep top m singular values to have sum of singular values > 0.99
    m1 = np.sum(np.cumsum(np.absolute(s1)) / np.sum(np.absolute(s1)) < 0.99)
    m2 = np.sum(np.cumsum(np.absolute(s2)) / np.sum(np.absolute(s2)) < 0.99)

    svacts1 = np.dot(s1[:m1]*np.eye(m1), V1[:m1])
    svacts2 = np.dot(s2[:m2]*np.eye(m2), V2[:m2])

    svcca_results = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=1e-10, verbose=False)
    svcca_cor = np.mean(svcca_results['cca_coef1'])

    return svcca_cor


def remove_dup(dat, dat_gene):
    # remove duplication of genes in dat_gene, only keep the first one, and remove the corresponding row in dat
    dat_gene_unique = []
    dat_unique = []
    for i, gene in enumerate(dat_gene):
        if gene not in dat_gene_unique:
            dat_gene_unique.append(gene)
            dat_unique.append(dat[i, :])
    dat_unique = np.stack(dat_unique, axis=0)
    return dat_unique, dat_gene_unique


def main():
    svCCA_cor_matrix = np.zeros((len(dat_names), len(dat_names)))
    svCCA_cor_adj_matrix = np.zeros((len(dat_names), len(dat_names)))
    svCCA_p_value_matrix = np.zeros((len(dat_names), len(dat_names)))

    for dat_i in range(len(dat_names)-1):
        for dat_j in tqdm(range(dat_i+1, len(dat_names))):
            dat1, dat2, dat1_name, dat2_name = process_dat(dat_i, dat_j, dat_names)
            ## svCCA
            if fit_svcca:
                svcca_cor, adj_svcca_cor, p_value = fit_svcca_model(dat1, dat2, permute=PERMUTE)
                svCCA_cor_matrix[dat_i, dat_j] = svcca_cor
                svCCA_cor_adj_matrix[dat_i, dat_j] = adj_svcca_cor
                svCCA_p_value_matrix[dat_i, dat_j] = p_value
    
    if fit_svcca:
        plot_cor_matrix(svCCA_cor_matrix, dat_names, 'svCCA')
        np.save(f'{RES_SAVE_PATH}/svCCA_cor_matrix.npy', svCCA_cor_matrix)
        if PERMUTE:
            plot_cor_matrix(svCCA_cor_adj_matrix, dat_names, 'svCCA_adj')
            np.save(f'{RES_SAVE_PATH}/svCCA_cor_adj_matrix.npy', svCCA_cor_adj_matrix)
            plot_cor_matrix(svCCA_p_value_matrix, dat_names, 'svCCA_p_value')
            np.save(f'{RES_SAVE_PATH}/svCCA_p_value_matrix.npy', svCCA_p_value_matrix)


if __name__ == '__main__':
    main()
