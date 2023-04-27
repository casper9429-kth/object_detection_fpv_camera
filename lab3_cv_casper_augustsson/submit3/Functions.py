import math
import numpy as np
import scipy
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.signal import convolve2d

#  SHOWGREY(IMAGE, DISPLAY, RESOLUTION, ZMIN, ZMAX)
#    displays the real matrix IMAGE as a gray-level image on the screen.
#
#  Arguments:
#    If RESOLUTION is a scalar value, RESOLUTION distinct gray-levels,
#    equidistantly spaced between 0 (black) and 1 (white), are used.
#    If RESOLUTION is a vector, its elements (which should lie in the 
#    interval [ 0, 1]) are used as gray-levels (colormap).
#
#    The (matrix element) values ZMIN and ZMAX are mapped to black and
#    white respectively. For a quantized image, it is often advisable 
#    to set ZMAX to the first quantization level outside actual range.
#    Values in ] ZMIN, ZMAX[ are mapped by linear interpolation, whereas
#    values outside this interval are mapped to either black or white by
#    truncation.
#
#    If ZMIN and ZMAX are omitted, they are set to the true max
#    and min values of the elements of IMAGE.
#
#    If RESOLUTION is also omitted, it is assumed to be 64.
#    If ZMIN = ZMAX, the displayed image is thresholded at ZMAX.
#
def showgrey(Image, display = True, res = 64, zmin = -1, zmax = -1):
	if zmin == -1:
		zmin = np.min(Image)
	if zmax == -1:
		zmax = np.max(Image)

	if not isinstance(res, list):
		col = np.linspace(0.0, 1.0, res)
	else:
		col = res
		res = len(col)

	range = 1.001*(zmax - zmin)
	if range == 0.0:
		range = 1e-10
	im = res*(Image.astype(np.float32) - zmin)/float(range)
	cols = np.array([ col, col, col ]).T
	cmap = ListedColormap(cols)
	plt.imshow(np.array(im, dtype=np.uint8), cmap)
	plt.axis('image')
	plt.axis('off')
	if display:
		plt.show()

#	SHOWFS( FREQSPEC, DISPLAY, RES) displays a compressed version
#	of the corresponding Fourier spectrum as a gray-level image.
#	RES distinct gray-levels, equidistantly spaced between 0 (black)
#	and 1 (white), are used. If RES is omitted, it is assumed to be 64.
#
def showfs(Image, display = True, res = 64):
	showgrey(np.log(1 + np.abs(np.fft.fftshift(Image))), display, res)


#	ROTIM = ROT( IMAGE, ANGLE, BKG) rotates the image IMAGE by an angle
#	ANGLE and sets the values of the background pixels to BKG.
#	If BKG is omitted, it is set to 0.
#	If ANGLE is also omitted, it is also set to 0.
#	This is not a very interesting rotation, but it will, like all
#	other values of ANGLE, create an output image ROTIM of size DxD,
#	where D = NORM( SIZE( IMAGE)), with the original image in the center.
#
def show(Image, display = True, res = 64):
	showgrey(np.log(1 + np.abs(Image)), display, res)

def rot(Image, angle = 0, bkg = 0):
	imgsz = np.shape(Image)
	diam = np.linalg.norm(imgsz)
	s = math.sin(angle*np.pi/180.0)
	c = math.cos(angle*np.pi/180.0)
	rad2 = (diam - 1)/2
	outsz = math.ceil(diam)
	gridvec = np.linspace(-rad2, rad2, outsz)
	X2, Y2 = np.meshgrid(gridvec, gridvec)
	X = np.round(c*X2 - s*Y2 + imgsz[1]/2).astype(int)
	Y = np.round(s*X2 + c*Y2 + imgsz[0]/2).astype(int)
	mask = np.where((X>=0) & (X<imgsz[1]) & (Y>=0) & (Y<imgsz[0]), 1, 0)
	X = mask*X
	Y = mask*Y
	Rotim = np.ndarray([outsz, outsz])
	Rotim = mask*Image[Y, X] + (1 - mask)*bkg
	return Rotim

# POW2IMAGE(inpic, threshold) -- Power spectrum as negative power of two
# POW2IMAGE performs a transformation in the Fourier domain such 
# that the phase information is preserved, whereas the magnitude 
# is REPLACED BY a power spectrum of the form
# |Fourier|^2 \sim 1/(a + |omega|^2)
#
def pow2image(inpic, a = 0.001):
	ftransform = np.fft.fft2(inpic)
	# Generate the power spectrum in centered frequency coordinates
	#(note that the factor (pi/umax) corresponds to (2*pi/usize))
	[usize, vsize] = np.shape(ftransform)
	umax = int(usize/2)
	vmax = int(vsize/2)
	[u, v] = np.meshgrid(range(umax - usize, umax), range(vmax - vsize, vmax))
	pow2spectrum = 1/(a + (np.pi*u/umax)**2 + (np.pi*v/vmax)**2)
	# Move the origin of the power spectrum to the lower left corner
	pow2spectrum = np.fft.fftshift(pow2spectrum)
	# Replace the power spectrum (NOT a linear operation)
	modtransform = pow2spectrum * np.exp(complex(0,1) * np.angle(ftransform))
	return np.real(np.fft.ifft2(modtransform))

def randphaseimage(inpic):
	ftransform = np.fft.fft2(inpic)
	[xsize, ysize] = np.shape(ftransform)
	phase = 2.0*np.pi*(np.random.rand(xsize, ysize) - 0.5)
	modtransform = np.abs(ftransform) * np.exp(complex(0,1) * phase)
	return np.real(np.fft.ifft2(modtransform))

# DELTAFCN(xsize, ysize) -- generates a discrete delta function of
# support xsize*ysize, in which the central pixel is set to one and 
# all other pixel values are set to zero.
#

def constampimage(inpic):
	ftransform = np.fft.fft2(inpic)
	# Generate the power spectrum in centered frequency coordinates
	#(note that the factor (pi/umax) corresponds to (2*pi/usize))
	modtransform = 10 * np.exp(complex(0,1) * np.angle(ftransform))
	return np.real(np.fft.ifft2(modtransform))

def deltafcn(xsize, ysize):
	matrix = np.zeros([xsize, ysize])
	matrix[(int(xsize/2), int(ysize/2))] = 1
	return matrix

def variance(inpic):
	[usize, vsize] = np.shape(inpic)
	[x, y] = np.meshgrid(np.linspace(-(usize-1)/2, (usize-1)/2, int(usize)), np.linspace(-(vsize-1)/2, (vsize-1)/2, int(vsize)))
	abspic = np.abs(inpic)
	norm = np.sum(abspic)
	ex = np.sum(abspic * x)/norm
	ey = np.sum(abspic * y)/norm
	exx = np.sum(abspic * x*x)/norm
	exy = np.sum(abspic * x*y)/norm
	eyy = np.sum(abspic * y*y)/norm
	mat = np.zeros([2, 2])
	mat[(0, 0)] = exx - ex*ex
	mat[(0, 1)] = exy - ex*ey
	mat[(1, 0)] = mat[(0, 1)]
	mat[(1, 1)] = eyy - ey*ey
	return mat

# DISCGAUSSFFT(pic, sigma2) -- Convolves an image by the
# (separable) discrete analogue of the Gaussian kernel by
# performing the convolution in the Fourier domain.
# The parameter SIGMA2 is the variance of the kernel.
# Reference: Lindeberg "Scale-space theory in computer vision", Kluwer, 1994.
#
def discgaussfft(inpic, sigma2):
	pfft = np.fft.fft2(inpic)
	[h, w] = np.shape(inpic)
	[x, y] = np.meshgrid(np.linspace(0, 1-1/w, w),np.linspace(0, 1-1/h, h))
	ffft = np.exp(sigma2 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y) - 2))
	print(np.shape(inpic), np.shape(ffft), np.shape(pfft))
	pixels = np.real(np.fft.ifft2(ffft * pfft))
	return pixels

# GAUSSNOISE(INPIC, SDEV, ZMIN, ZMAX) adds white (uncorrelated)
# Gaussian noise with standard deviation SDEV to INPIC.
# If the arguments ZMIN and ZMAX are specified and are unequal, 
# the output values are truncated to the range [ZMIN, ZMAX]
#
def gaussnoise(inpic, sdev, zmin = 0, zmax = 0):
	noisy = inpic + np.random.normal(0, sdev, np.shape(inpic))
	if zmin < zmax:
		noisy = np.max(zmin, np.min(zmax, noisy))
	return noisy

# SAPNOISE( inpic, FRAC, ZMIN, ZMAX) adds salt-and-peppar noise to an
# image by resetting a fraction FRAC/2 of the pixels to ZMIN and a 
# similar fraction to ZMAX in a pixel-to-pixel independent manner.
# If ZMIN and ZMAX are omitted, they are set to the true minimum and
# maximum values of inpic.
#
def sapnoise(inpic, frac, zmin = 1, zmax = 0):
	if zmin > zmax:
		zmin = np.min(inpic)
		zmax = np.max(inpic)
	noisy = inpic
	[u, v] = np.shape(inpic)
	Rand = np.random.rand(u, v)
	index = np.where(Rand < frac/2)
	noisy[index] = zmin 
	index = np.where(Rand > 1 - frac/2)
	noisy[index] = zmax 
	return noisy

# MEDIM = MEDFILT( IMAGE, WHEIGHT, WWIDTH) computes an output image
# MEDIM by applying a median filter with window height WHEIGHT and
# window width WWIDTH to the input image IMAGE.
# If WWIDTH is omitted, it is set to WHEIGHT.
#
def medfilt(Image, wheight, wwidth = -1):
	if wwidth == -1:
		wwidth = wheight
	result = scipy.ndimage.median_filter(Image, size=(wheight, wwidth))
	return result


# FILTIM = IDEAL( IMAGE, CUTOFF, FTYPE) filters the image IMAGE with
# an ideal high-pass or low-pass filter with cut-off frequency CUTOFF
# cycles per pixel and returns the resulting image FILTIM along with
# the modulation transfer function MTF.
# The filter is high-pass or low-pass depending on whether FTYPE = 'h'
# or FTYPE = 'l' respectively. If FTYPE is omitted, it is set to 'l'.
#
def ideal(Image, cutoff, ftype = 'l'):
	[u, v] = np.shape(Image)
	ur = np.linspace(-u/2, u/2-1, u)/u
	vr = np.linspace(-v/2, v/2-1, v)/v
	x, y = np.meshgrid(ur, vr)
	cutoff2 = cutoff * cutoff
	MTF = np.where(x**2 + y**2 > cutoff2, 0, 1)
	if ftype == 'h':
		MTF = 1 - MTF
	Filtim = np.real(np.fft.ifft2(np.fft.fftshift(MTF) * np.fft.fft2(Image)))
	return Filtim

# RAWSUBSAMPLE -- reduce image size by raw subsampling without presmoothing
# rawsubsample(image) reduces the size of an image by a factor of two in
# each dimension by raw subsampling, i.e., by picking out every second
# pixel along each dimension.
#
def rawsubsample(inpic):
	[h, w] = np.shape(inpic)
	[h2, w2] = [int(h/2), int(w/2)]
	newimg = np.zeros((h2, w2))
	ur = np.linspace(0, h2*2-2, h2)
	vr = np.linspace(0, w2*2-2, h2)
	[X, Y] = np.meshgrid(ur, vr)
	newimg = inpic[Y.astype(int), X.astype(int)]
	return newimg

# BINSUBSAMPLE -- subsampling with binomial presmoothing
#   binsubsample(image) reduces the size of an image by first smoothing
#   it with a two-dimensional binomial kernel having filter coefficients 
#      (1/16  1/8  1/16)
#      ( 1/8  1/4  1/8)
#      (1/16  1/8  1/16)
#   and then subsampling it by a factor of two in each dimension.
def binsubsample(inpic):
	prefilterrow = np.array([[0.25, 0.50, 0.25]])
	prefilter = prefilterrow.T @ prefilterrow
	print(prefilter)
	presmoothpic = convolve2d(inpic, prefilter, 'same')
	pixels = rawsubsample(presmoothpic)
	return pixels

# MASK = CONTOUR( IMG) finds zero-crossings in an image by looking at the sign
# of the four naighbours and returns the zero-crossing points in MASK
#
def contour(img):
	[w, h] = np.shape(img)
	[x, y] = np.meshgrid(range(0, w), range(0, h))
	imgu = img[np.maximum(y-1, 0), x]
	imgd = img[np.minimum(y+1, h-1), x]
	imgl = img[y, np.maximum(x-1, 0)]
	imgr = img[y, np.minimum(x+1, w-1)]
	mask = ((img*imgu <= 0) & (np.abs(img)<np.abs(imgu)) |
		(img*imgl <= 0) & (np.abs(img)<np.abs(imgl)) |
		(img*imgr <= 0) & (np.abs(img)<np.abs(imgr)) |
		(img*imgd <= 0) & (np.abs(img)<np.abs(imgd))).astype(int)
	return mask
		
# ZEROCROSSCURVES(ZERO, MASK) -- Extraction of zero-crossing curves
# Computes the zero-crossing curves from the image ZERO
# If the MASK image is specified, only point on the zero-crossing
# curves for which the mask value is True are reserved
# The format of these curves is a tuple of two arrays with (Y,X) coordinates.
#
def zerocrosscurves(zeropic, mask):
	curves = np.where(contour(zeropic) & mask.astype(int))
	return curves

# THRESHOLDCURVES(CURVES, MASK) -- Thresholding of curves
# Returns a new set of curves containing only those points
# in the polygons for which the mask value is True.
# The format of these curves is a tuple of two arrays with (Y,X) coordinates.
#
def thresholdcurves(curves, mask):
	(Y, X) = curves
	m = mask[Y, X]
	X = X[m]
	Y = Y[m]
	return (Y, X)

# OVERLAYCURVES(IMAGE, CURVES)
# Displays CURVES overlayed on IMAGE
# The format of these curves is a tuple of two arrays with (Y,X) coordinates.
#
def overlaycurves(img, curves):
	[h, w] = np.shape(img)
	rgb = np.zeros((h, w, 3), dtype=np.uint8)
	rgb[:,:,0] = rgb[:,:,1] = rgb[:,:,2] = img
	(Y, X) = curves
	rgb[Y, X, 0] = 0
	rgb[Y, X, 1] = 0
	rgb[Y, X, 2] = 255
	plt.imshow(rgb)
	plt.axis('image')
	plt.axis('off')

# [ POS, VALUE, ANMS] = LOCMAX8( A) finds the 8-connectedness local
# maxima of the matrix A and returns a array POS with the positions
# of the local maxima, a list VALUE with the corresponding local
# maximal values, and a matrix ANMS equal to A at the local maxima
# and set to 0 everywhere else.
#
def locmax8(A):
	[h, w] = np.shape(A)
	Aexp = np.zeros((h + 2, w + 2))
	Aexp[1:h + 1, 1:w + 1] = A
	Anms = np.ones(np.shape(A)).astype(bool)
	Anms = Anms & (A >= Aexp[0:h + 0, 0:w + 0])
	Anms = Anms & (A >= Aexp[0:h + 0, 1:w + 1])
	Anms = Anms & (A >= Aexp[0:h + 0, 2:w + 2])
	Anms = Anms & (A >= Aexp[1:h + 1, 0:w + 0])
	Anms = Anms & (A >  Aexp[1:h + 1, 2:w + 2])
	Anms = Anms & (A >  Aexp[2:h + 2, 0:w + 0])
	Anms = Anms & (A >  Aexp[2:h + 2, 1:w + 1])
	Anms = Anms & (A >  Aexp[2:h + 2, 2:w + 2])
	[Y, X] = np.where(Anms)
	Pos = np.concatenate((X.reshape(-1,1), Y.reshape(-1,1)), axis=1)
	Value = A[Y, X]
	Anms = np.zeros(np.shape(A))
	Anms[Y, X] = Value
	return Pos, Value, Anms

# IMEAN = MEAN_SEGMENTS(I, SEGM) computes the mean within clusters indexed per
# pixel by SEGM and returns in image with pixel values replaced by the means
#
def mean_segments(I, segm):
	Ivec = np.reshape(I, (-1, 3))
	idx = np.reshape(segm, (-1))
	L = np.max(idx) + 1
	centers = np.zeros((L, 3))
	h, _ = np.histogram(idx, L, (-0.5, L-0.5))
	for l in range(L):
		if h[l] > 0:
			centers[l,:] = np.mean(Ivec[idx == l], axis=0)
	return centers[segm,:]

# INEW = OVERLAY_BOUNDS(I, SEGM) creates a new image INEW with boundaries
# between clusters as indicated by the segmentation SEGM
#
def overlay_bounds(I, segm):
	(h, w) = np.shape(segm)
	grow = np.zeros((h + 1, w + 1), dtype = segm.dtype)
	grow[0:h,0:w] = segm
	mask = (segm != grow[1:h+1, 0:w]) | (segm != grow[0:h, 1:w+1])  | (segm != grow[1:h+1, 1:w+1]) 
	mask[0:h, w-1] = mask[h-1, 0:w] = False
	Inew = np.array(I)
	Inew[mask, 0] = 255
	Inew[mask, 1] = Inew[mask, 2] = 0
	return Inew
