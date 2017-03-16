
from __future__ import print_function
from __future__ import division

import glob
import gdal
import numpy as np
import os
import scipy.misc
import joblib		# (we could use pickle instead if you wish)

# summer and winter data is assumed to be in the ./data folder
# (in their own folders)

# for recognition, images should (generally) be square,
# specify a size
# TODO: read from config file instead so this is portable
# don't make this too big or you'll run out of memory!
img_dim = 100

# only needed for debugging scaling and data preprocessing below,
# will output images to ./images/<obj class> folders
save_imgs = True

# rescale images? will distort, might not be desirable
rescale = False

if not os.path.exists("./data/images") and save_imgs:
	os.mkdir("./data/images")

summer_data = []
winter_data = []
count = 0

geotifs = glob.glob("./data/*/*.tif")

for tif_path in geotifs:
	obj_class = tif_path.split("\\")[-2]
	filename = tif_path.split("\\")[-1][:-4]
	
	src_ds = gdal.Open(tif_path)

	# get nodata values
	# assume not unique per channel
	red_no_data_mask = src_ds.GetRasterBand(1).GetNoDataValue()
	blue_no_data_mask = src_ds.GetRasterBand(2).GetNoDataValue()
	green_no_data_mask = src_ds.GetRasterBand(3).GetNoDataValue()

	red = src_ds.GetRasterBand(1).ReadAsArray()
	blue = src_ds.GetRasterBand(2).ReadAsArray()
	green = src_ds.GetRasterBand(3).ReadAsArray()

	# set no datas to zeros
	red[red == red_no_data_mask] = 0
	blue[blue == blue_no_data_mask] = 0
	green[green == green_no_data_mask] = 0

	# ensure image shape is even on both sides
	remainder = [red.shape[0] % 2, red.shape[1] % 2]

	img = np.zeros((red.shape[0] + remainder[0], red.shape[1] + remainder[1], 3), dtype = np.int32)

	img[:red.shape[0], :red.shape[1], 0] = red
	img[:red.shape[0], :red.shape[1], 1] = blue
	img[:red.shape[0], :red.shape[1], 2] = green

	# lay image on its side so longest dimension is along the width
	if img.shape[0] > img.shape[1]:
		img = np.transpose(img, (1, 0, 2))

	# check both dimensions to see if we need to pad,
	# so that image is rougly centered
	if img.shape[0] < img_dim and img.shape[1] < img_dim:
		padded = np.zeros((img_dim, img_dim, 3), dtype = np.int32)
		half_width = img.shape[1]//2
		half_height = img.shape[0]//2

		padded[img_dim//2 - half_height:img_dim//2 + half_height, img_dim//2 - half_width:img_dim//2 + half_width, :] = img

		img = padded

	# pad along height dimension, ensuring that it is centered along the middle
	elif img.shape[0] < img_dim:
		padded = np.zeros((img_dim, img.shape[1], 3), dtype = np.int32)
		half_height = img.shape[0]//2
		padded[img_dim//2 - half_height:img_dim//2 + half_height, :, :] = img
		img = padded
	
	# pad along width dimension, centering again
	elif img.shape[1] < img_dim:
		padded = np.zeros((img.shape[0], img_dim, 3), dtype = np.int32)
		half_width = img.shape[1]//2
		padded[:, img_dim//2 - half_width:img_dim//2 + half_width, :, :] = img
		img = padded

	
	# check to see if image is bigger than img_dim x img_dim, if so
	# extract img_dim x img_dim patches
	# note that anything greater than a multiple of img_dim is not
	# captured
	img_shape = img.shape
	if img_shape[0] > img_dim or img_shape[1] > img_dim:
		patches = []
		height_excess = img_shape[0] - img_dim
		width_excess = img_shape[1] - img_dim

		n_patches_h = 1
		n_patches_w = 1
		# find out how many patches fit in the image
		if height_excess > 0:
			n_patches_h = img_shape[0] // img_dim
		if width_excess > 0:
			n_patches_w = img_shape[1] // img_dim
		for i in range(1, n_patches_h+1):
			for j in range(1, n_patches_w+1):
				p = img[img_dim*(i-1):img_dim*(i), img_dim*(j-1):img_dim*(j)]
				
				# don't collect empty patches
				if np.sum(p) > 0:
					patches.append(p)
		
		if len(patches) > 0:
			if save_imgs:
				if not os.path.exists("./data/images/" + obj_class):
					os.mkdir("./data/images/" + obj_class)
				for patch in patches:
					scipy.misc.imsave("./data/images/" + obj_class + "/" + str(count) + ".png", patch)
					count += 1
			for patch in patches:
				if obj_class == "summer":
					summer_data.append(img)
				else:
					winter_data.append(img)
			continue



	# resize to square
	if rescale:
		img = scipy.misc.imresize(img, (img_dim, img_dim))
	
	if save_imgs:
		scipy.misc.imsave("./data/images/" + obj_class + "/" + filename + ".png", img)

	if obj_class == "summer":
		summer_data.append(img)
	else:
		winter_data.append(img)

summer_data = np.array(summer_data)
winter_data = np.array(winter_data)

# TODO: this is a good place to do a train/test split for validation by others

joblib.dump(summer_data, "./data/converted_summer.joblib")
joblib.dump(winter_data, "./data/converted_winter.joblib")

