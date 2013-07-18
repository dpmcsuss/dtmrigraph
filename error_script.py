from brain import *
from lcc import ConnectedComponent
from svd import *
from numpy import *
#from pyflann import * 
from apprknn import get_cv_neighbors

'''
This program takes as input the key of a .mat file (ie 'M817....") and returns a vector of the errors associated to the cross-validation for dimensions in multiples of 10. 
'''

#loads fibergraph & associated adjacency matrix
bfn = input('Enter the key of the .mat file you want to embed: ')
G = load_fibergraph(roiDir, graphDir, bfn)
print('Reading key...')
G_matrix = G.spcscmat
print('Obtaining largest component...')

#finds largest connected component 
lconn = ConnectedComponent(G_matrix)
lconn.save('/data/projects/MRN/roiknn/' + bfn)
fileName = '/mnt/braingraph2data/projects/MR/MRN/graphs/biggraphs/' + bfn + '_fiber.mat'
outFile = '/data/projects/MRN/roiknn/' + bfn + '_embed.npy'
lconnPath = '/data/projects/MRN/roiknn/' + bfn + '_concomp.npy'
n = int(input('What dimension would you like to embed to?: '))
print('Embedding graph...')

#embeds graph
embed_graph(lconnPath, fileName, outFile, n)
print('The embedding is located in' + outFile)
lccDir = '/data/projects/MRN/roiknn/'
embedDir = lccDir
X, ccRoi, G = get_stfp_data_from_fn(roiDir, lccDir, embedDir, graphDir, bfn)
cv_dim = 10
error_vector = []

#finds cross validation errors 
while cv_dim <= n:
        print('Obtaining cross validation neighbors...')
        neighN, neighC = get_cv_neighbors(X, ccRoi, G, 5, 1, cv_dim)
        error = mean(not_equal(ccRoi, ccRoi[neighN]))
        error_vector = error_vector + [error]
        cv_dim = cv_dim + 10
        if cv_dim > n:
                break
print(error_vector)
~                      
