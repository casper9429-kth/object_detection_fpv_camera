import sys
import math
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from lab3 import kmeans_segm
from Functions import showgrey, mean_segments, overlay_bounds
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import distance_matrix
from scipy import sparse, optimize


def ncuts(A, D, n_ev):
    # Computes the n_ev smallest (non-zero) eigenvectors and eigenvalues of the 
    # of the Laplacian of A, where D is diagonal matrix with the row sums of A
    L = (D - A) + 1e-3*sparse.identity(np.size(D, 0))
    success = True
    try:
        #EVal, EV = sparse.linalg.lobpcg(L, X, M=D, largest=False)
        EVal, EV = sparse.linalg.eigsh(L, k=n_ev, M=D, which='SM', tol=1e-3, maxiter=10000)
    except:
        print("Failed to find eigenvectors")
        EVal = np.array([ 0, 0 ])
        EV = np.zeros((np.size(L, axis=0), 2))
        success = False
    return EV, EVal, success


# NcutValue - 2.1 Computing the Optimal Partition Ncut. eq (5)
#
# Synopsis
#  ncut = ncuts_value(T, U2, D, W);
#
# Inputs ([]s are optional)
#  (scalar) t        splitting point (threshold)
#  (vector) U2       N x 1 vector representing the 2nd smallest
#                     eigenvector computed at step 2.
#  (matrix) W        N x N weight matrix
#  (matrix) D        N x N diagonal matrix
#
# Outputs ([]s are optional)
#  (scalar) ncut     The value calculated at the right term of eq (5).
#                    This is used to find minimum Ncut.
#
# Authors
#  Naotoshi Seo <sonots(at)sonots.com>
#
# License
#  The program is free to use for non-commercial academic purposes,
#  but for course works, you must understand what is going inside to use.
#  The program can be used, modified, or re-distributed for any purposes
#  if you or one of your group understand codes (the one must come to
#  court if court cases occur.) Please contact the authors if you are
#  interested in using the program without meeting the above conditions.
#
# Changes
#  10/01/2006  First Edition
#  15/10/2021  Python version (Marten Bjorkman)

def ncuts_value(t, U2, W, D):
    x = np.where(U2 > t, 1, -1)
    d = D.diagonal()
    k = np.sum(d[x > 0]) / sum(d)
    b = k/(1 - k)
    y = (1 + x) - b*(1 - x)
    ncut = y.dot((D - W) @ y) / y.dot(D @ y)
    return ncut


# NcutPartition - Partitioning
#
# Synopsis
#  [sub ids ncuts] = ncuts_partition(I, W, sNcut, sArea, [id])
#
# Description
#  Partitioning. This function is called recursively.
#
# Inputs ([]s are optional)
#  (vector) I        N x 1 vector representing a segment to be partitioned.
#                    Each element has a node index of V (global segment).
#  (matrux) W        N x N matrix representing the computed similarity
#                    (weight) matrix.
#                    W(i,j) is similarity between node i and j.
#  (scalar) sNcut    The smallest Ncut value (threshold) to keep partitioning.
#  (scalar) sArea    The smallest size of area (threshold) to be accepted
#                    as a segment.
#  (string) [id]     A label of the segment (for debugg)
#
# Outputs ([]s are optional)
#  (cell)   Seg      A cell array of segments partitioned.
#                    Each cell is the each segment.
#  (cell)   Id       A cell array of strings representing labels of each segment.
#                    IDs are generated as children based on a parent id.
#  (cell)   Ncut     A cell array of scalars representing Ncut values
#                    of each segment.
#
# Requirements
#  NcutValue
#
# Authors
#  Naotoshi Seo <sonots(at)sonots.com>
#
# License
#  The program is free to use for non-commercial academic purposes,
#  but for course works, you must understand what is going inside to use.
#  The program can be used, modified, or re-distributed for any purposes
#  if you or one of your group understand codes (the one must come to
#  court if court cases occur.) Please contact the authors if you are
#  interested in using the program without meeting the above conditions.
#
# Changes
#  10/01/2006  First Edition
#  15/10/2021  Python version (Marten Bjorkman)

def ncuts_partition(I, W, sNcut, sArea, id, maxDepth, depth):
    N = np.shape(W)[0]
    d = np.sum(W, axis=1)
    D = sparse.spdiags(d.reshape(-1), [0], N, N, format='csr') # D = diagonal matrix

    # Step 2 and 3. Solve generalized eigensystem (D -W)*S = S*D*U (12).
    # (13) is not necessary thanks to smart matlab. Get the 2 smallests ('sm')
    [EV, EVal, success] = ncuts(W, D, 2)

    # 2nd smallest (1st smallest has all same value elements, and useless)
    U2 = EV[:, 1]
    
    # Step 3. Refer 3.1 Example 3.
    # Bipartition the graph at point that Ncut is minimized.
    t = np.mean(U2)
    if success == True:
        t = optimize.minimize(ncuts_value, t, args=(U2, W, D), method='Nelder-Mead', options={'maxiter': 20}).x
    A = np.where(U2 > t)[0]
    B = np.where(U2 <= t)[0]
    
    # Step 4. Decide if the current partition should be divided
    #   if either of partition is too small, stop recursion.
    #   if Ncut is larger than threshold, stop recursion.
    if success == True:
        ncut = ncuts_value(t, U2, W, D)
    else:
        ncut = sNcut
    print(f'Cutting ncut=%.3f sizes=(%d,%d) %s' % (ncut, np.size(A), np.size(B), id))
    if np.size(A)<sArea or np.size(B)<sArea or ncut>=sNcut or depth>maxDepth:
        Seg = [ I ]
        Id = [ id ]     # for debugging
        Ncut = [ ncut ] # for debugging
        return Seg, Id, Ncut

    # recursively create segments of A
    SegA, IdA, NcutA = ncuts_partition(I[A], W[:,A][A,:], sNcut, sArea, id+'-A', maxDepth, depth+1)

    # recursively create segments of B
    SegB, IdB, NcutB = ncuts_partition(I[B], W[:,B][B,:], sNcut, sArea, id+'-B', maxDepth, depth+1)
    
    # concatenate cell arrays
    Seg  =  SegA + SegB 
    Id   =  IdA + IdB
    Ncut =  NcutA + NcutB
    return Seg, Id, Ncut


def ncuts_affinity(im, XY_RADIUS, RGB_SIGMA):
    (h, w, _) = np.shape(im)

    # Find all pairs of pixels within a distance of XY_RADIUS
    rad = int(math.ceil(XY_RADIUS))
    [di,dj] = np.meshgrid(range(-rad, rad + 1), range(-rad, rad + 1))
    dv = (dj**2 + di**2) <= XY_RADIUS**2
    di = di[dv]
    dj = dj[dv]
    [i,j] = np.meshgrid(range(w), range(h))
    i = np.repeat(i[:, :, np.newaxis], len(di), axis=2)
    j = np.repeat(j[:, :, np.newaxis], len(di), axis=2)
    i_ = i - di
    j_ = j - dj
    v = np.where((i_ >= 0) & (i_ < w) & (j_ >= 0) & (j_ < h))
    pair_i =  j[v]*w +  i[v]
    pair_j = j_[v]*w + i_[v]

    # Weight each pair by the difference in RGB values, divided by RGB_SIGMA
    RGB = np.reshape(im/RGB_SIGMA, (-1, 3))
    R = RGB[pair_i,:]
    W = np.exp(-np.sum((RGB[pair_i,:] - RGB[pair_j,:])**2, axis=1)).astype(np.float64)
    
    # Construct an affinity matrix
    A = sparse.csr_matrix((W, (pair_i, pair_j)), shape=(w*h, w*h))

    return A


def norm_cuts_segm(I, colour_bandwidth, radius, ncuts_thresh, min_area, max_depth):
    
    (nRow, nCol, c) = np.shape(I)
    N = nRow * nCol
    V = np.reshape(I, (N, c))    
    
    print('Compute affinity matrix...')
    W = ncuts_affinity(I, radius, colour_bandwidth)

    print('Solve eigenvalue problems to find partitions...')
    Seg = np.arange(N, dtype=np.int32)
    [Seg, Id, Ncut] = ncuts_partition(Seg, W, ncuts_thresh, min_area, 'ROOT', max_depth, 1)

    segm = np.zeros((N, 1), dtype=np.int32)
    for i in range(len(Seg)):
        segm[Seg[i]] = i
        print('Ncut = %f  %s' % (Ncut[i], Id[i]))
        
    segm = np.reshape(segm, (nRow, nCol)).astype(np.int32)
    return segm



############################################

def norm_cuts_example():
    colour_bandwidth = 20.0  # color bandwidth
    radius = 20 #10         # maximum neighbourhood distance
    ncuts_thresh = 0.3 #0.3  # cutting threshold
    min_area = 150           # minimum area of segment
    max_depth = 1           # maximum splitting depth
    scale_factor = 0.25      # image downscale factor
    image_sigma = 0.5        # image preblurring scale

    img = Image.open('Images-jpg/tiger1.jpg')
    img = img.resize((int(img.size[0]*scale_factor), int(img.size[1]*scale_factor)))
     
    h = ImageFilter.GaussianBlur(image_sigma)
    I = np.asarray(img.filter(ImageFilter.GaussianBlur(image_sigma))).astype(np.float32)
    
    segm = norm_cuts_segm(I, colour_bandwidth, radius, ncuts_thresh, min_area, max_depth)
    Inew = mean_segments(img, segm)
    if True:
        Inew = overlay_bounds(img, segm)

    img = Image.fromarray(Inew.astype(np.ubyte))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    img.save('result/normcuts1.png')

if __name__ == '__main__':
    sys.exit(norm_cuts_example())

