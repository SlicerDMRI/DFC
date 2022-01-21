import io
import numpy as np
import vtk
import whitematteranalysis as wma
from utils import fibers

import scipy
import numpy.ma as ma

def feat_RAS(pd_tract, number_of_points=15,bilateral=False):
	
	"""The most simple feature for initial test

	Parameters
	----------
	TODO:

	"""
	fiber_array = fibers.FiberArray()
	fiber_array.convert_from_polydata(pd_tract, points_per_fiber=number_of_points)
	if bilateral is True:
		feat = np.dstack((abs(fiber_array.fiber_array_r), fiber_array.fiber_array_a,fiber_array.fiber_array_s))
		featf = np.dstack((abs(fiber_array.fiber_array_r[:,::-1]), fiber_array.fiber_array_a[:,::-1], fiber_array.fiber_array_s[:,::-1]))
	else:
		feat = np.dstack((fiber_array.fiber_array_r, fiber_array.fiber_array_a, fiber_array.fiber_array_s))
	featp=fiber_array.fiber_array_p
	roi_list=fiber_array.roi_list
	return feat,featf,roi_list

def feat_orientation(pd_tract, number_of_points=15):
	
	"""The most simple feature for initial test

	Parameters
	----------
	TODO:

	"""

	feat_RAS_ = feat_RAS(pd_tract, number_of_points=number_of_points)

	margin_zeros = np.zeros((feat_RAS_.shape[0], 1, feat_RAS_.shape[2]))

	# the last point does not have Next
	diff_to_next = np.concatenate((feat_RAS_[:, :-1, :]-feat_RAS_[:, 1:, :], margin_zeros), axis=1)
	# the first point does not have Previous
	diff_to_prev = np.concatenate((margin_zeros, feat_RAS_[:, 1:, :]-feat_RAS_[:, :-1, :]), axis=1)

	norm_diff_to_next = np.sqrt(np.sum(np.square(diff_to_next), axis=2))
	norm_diff_to_prev = np.sqrt(np.sum(np.square(diff_to_prev), axis=2))
	
	# normalize
	for idx in range(diff_to_next.shape[2]):

		diff_to_next[..., idx] = diff_to_next[..., idx] / norm_diff_to_next

		diff_to_prev[..., idx] = diff_to_prev[..., idx] / norm_diff_to_prev

	feat = np.concatenate((feat_RAS_, diff_to_prev, diff_to_next), axis=2)
	return feat 

def feat_curv_tors(pd_tract, number_of_points=15):
	
	"""The most simple feature for initial test

	Parameters
	----------
	TODO:

	"""

	fiber_array = fibers.FiberArray()
	fiber_array.convert_from_polydata_with_trafic(pd_tract, points_per_fiber=number_of_points)


	feat = np.dstack((fiber_array.fiber_array_cur, fiber_array.fiber_array_tor))
	
	return feat 

def feat_RAS_curv_tors(pd_tract, number_of_points=15):
	
	"""The most simple feature for initial test

	Parameters
	----------
	TODO:

	"""

	fiber_array = fibers.FiberArray()
	fiber_array.convert_from_polydata_with_trafic(pd_tract, points_per_fiber=number_of_points)


	feat = np.dstack((fiber_array.fiber_array_r, fiber_array.fiber_array_a, fiber_array.fiber_array_s, fiber_array.fiber_array_cur, fiber_array.fiber_array_tor))
	
	return feat 

def feat_RASF(pd_tract, number_of_points=15):
	
	"""The most simple feature for initial test

	Parameters
	----------
	TODO:

	"""

	fiber_array = fibers.FiberArray()
	fiber_array.convert_from_polydata_with_FS(pd_tract, points_per_fiber=number_of_points)

	fiber_array_fs_t = _transform_fs_labels(fiber_array.fiber_array_fs)

	feat = np.dstack((fiber_array.fiber_array_r, fiber_array.fiber_array_a, fiber_array.fiber_array_s, fiber_array_fs_t))
	
	return feat

def feat_RAS_3D(pd_tract, number_of_points=15, repeat_time=15,bilateral=False):
	
	"""The most simple feature for initial test

	Parameters
	----------
	TODO:

	"""

	feat,featf,featp= feat_RAS(pd_tract, number_of_points=number_of_points,bilateral=bilateral)
	#featp=np.expand_dims(featp, axis=2)
	feat_1221_2112_repeat = _feat_to_3D(feat, repeat_time=repeat_time)
	feat_1221_2112_repeatf = _feat_to_3D(featf, repeat_time=repeat_time)
	#feat_1221_2112_repeatp = _feat_to_3D(featp, repeat_time=repeat_time)
	#feat_com=np.concatenate((feat_1221_2112_repeat,feat_1221_2112_repeatf),axis=-1)
	feat_array = feat.reshape(feat.shape[0], -1)
	return feat_1221_2112_repeat,feat_array,feat_1221_2112_repeatf,featp

def feat_RASF_3D(pd_tract, number_of_points=15, repeat_time=15):
	
	"""The most simple feature for initial test

	Parameters
	----------
	TODO:

	"""

	feat = feat_RASF(pd_tract, number_of_points=number_of_points)

	feat_1221_2112_repeat = _feat_to_3D(feat, repeat_time=repeat_time)

	return feat_1221_2112_repeat

def feat_orientation_3D(pd_tract, number_of_points=15, repeat_time=15):
	
	"""The most simple feature for initial test

	Parameters
	----------
	TODO:

	"""

	feat = feat_orientation(pd_tract, number_of_points=number_of_points)

	feat_1221_2112_repeat = _feat_to_3D(feat, repeat_time=repeat_time)

	return feat_1221_2112_repeat

def feat_1D(pd_tract, number_of_points=15):
	
	"""The most simple feature for initial test

	Parameters
	----------
	TODO:

	"""

	fiber_array = wma.fibers.FiberArray()
	fiber_array.convert_from_polydata(pd_tract, points_per_fiber=number_of_points)

	feat = np.concatenate((fiber_array.fiber_array_r, fiber_array.fiber_array_a, fiber_array.fiber_array_s), axis=1)

	return feat


## internal function
def _feat_to_3D(feat, repeat_time=15):

	# 1 first; 2 last
	# 12 is the original point order
	# 21 is the fliped point order
	feat_12 = feat
	feat_21 = np.flip(feat_12, axis=1)

	# concatenate the different orders 
	feat_1221 = np.concatenate((feat_12, feat_21), axis=1)
	feat_2112 = np.concatenate((feat_21, feat_12), axis=1)

	# reshape to a 4D array
	feat_shape = (feat_1221.shape[0], 1, feat_1221.shape[1], feat_1221.shape[2])
	
	feat_1221 = np.reshape(feat_1221, feat_shape)
	feat_2112 = np.reshape(feat_2112, feat_shape)

	# Now the dimension is (# of fibers, 2, # of points, 3)
	# the second D is [1221; 2112]; the fourth D is RAS
	feat_1221_2112 = np.concatenate((feat_1221, feat_2112), axis=1)

	# Repeat the send D;
	# In the tmp variable, it is [1221; 1221; ...; 2112; 2112; ....], 
	# but we want [1221; 2112; 1221; 2112; ....]
	feat_1221_2112_repeat_tmp = np.repeat(feat_1221_2112, repeat_time, axis=1)
	#del feat_1221_2112
	feat_1221_2112_repeat = np.zeros(feat_1221_2112_repeat_tmp.shape)

	feat_1221_2112_repeat[:, 0::2, :, :] = feat_1221_2112_repeat_tmp[:, 0:repeat_time, :, :]
	feat_1221_2112_repeat[:, 1::2, :, :] = feat_1221_2112_repeat_tmp[:, repeat_time:, :, :]

	return feat_1221_2112_repeat

def _transform_fs_labels(fs_array):

	if 0:
		unique_fs_labels = np.sort(np.unique(fs_array).astype(int))
		fs_dict = {region:i for i, region in enumerate(unique_fs_labels)}
		np.save('fs_dict.npy', fs_dict)

		fs_dict = np.load('fs_dict.npy')
	
	unique_fs_labels = np.sort(np.unique(fs_array).astype(int))

	fs_array_t = fs_array
	for fs_label in unique_fs_labels:
		fs_array_t[fs_array==fs_label] = _region_label_LR(fs_label)
		
	return fs_array_t

def _region_label_LR(label):

    CC_regions = range(251, 256)
    CC_regions_new = np.arange(1, 1+len(CC_regions))

    commissural_sub_cortical_regions = [14, 15, 16, 24, 72, 77, 80, 85]
    commissural_sub_cortical_regions_new = np.arange(CC_regions_new[-1]+1, CC_regions_new[-1]+1+len(commissural_sub_cortical_regions))

    # combine left and right
    left_sub_cortical_regions = range(1, 14) + range(17, 21) + range(25, 40) # [2, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 30, 31]
    left_sub_cortical_regions_new = np.arange(commissural_sub_cortical_regions_new[-1], commissural_sub_cortical_regions_new[-1]+1+len(left_sub_cortical_regions))

    right_sub_cortical_regions = range(40, 72) # [41, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62, 63]
    right_sub_cortical_regions_new = - left_sub_cortical_regions_new

    left_GM_cortical_regions = range(1000, 1036)
    left_GM_cortical_regions_new = np.arange(left_sub_cortical_regions_new[-1]+1, left_sub_cortical_regions_new[-1]+1+len(left_GM_cortical_regions))

    right_GM_cortical_regions = range(2000, 2036)
    right_GM_cortical_regions_new = - left_GM_cortical_regions_new

    left_WM_cortical_regions = range(3001, 3036)
    left_WM_cortical_regions_new = np.arange(left_GM_cortical_regions_new[-1]+1, left_GM_cortical_regions_new[-1]+1+len(left_WM_cortical_regions))

    right_WM_cortical_regions = range(4001, 4036)
    right_WM_cortical_regions_new = - left_WM_cortical_regions_new

    WM_Unsegmented = [5001, 5002]
    WM_Unsegmented_new = np.array([-120, 120])

    if label in right_WM_cortical_regions:
        label = right_WM_cortical_regions_new[np.where(right_WM_cortical_regions == label)[0]]
    elif label in left_WM_cortical_regions:
        label = left_WM_cortical_regions_new[np.where(left_WM_cortical_regions == label)[0]]
    elif label in right_GM_cortical_regions:
        label = right_GM_cortical_regions_new[np.where(right_GM_cortical_regions == label)[0]]
    elif label in left_GM_cortical_regions:
        label = left_GM_cortical_regions_new[np.where(left_GM_cortical_regions == label)[0]]
    elif label in CC_regions:
        label = CC_regions_new[np.where(CC_regions == label)[0]]
    elif label in commissural_sub_cortical_regions:
        label = commissural_sub_cortical_regions_new[np.where(commissural_sub_cortical_regions == label)[0]]
    elif label in right_sub_cortical_regions:
        label = right_sub_cortical_regions_new[np.where(right_sub_cortical_regions == label)[0]]
    elif label in left_sub_cortical_regions:
        label = left_sub_cortical_regions_new[np.where(left_sub_cortical_regions == label)[0]]
    elif label in WM_Unsegmented:
        label = WM_Unsegmented_new[np.where(WM_Unsegmented == label)[0]]
    else:
        label = np.array(0)

    return -label

##
def downsample(ds_step, x_data, y_data=None):
	x_data_ds = x_data[::ds_step, :, :, :]
	
	y_data_ds = None
	if y_data is not None:
		y_data_ds = y_data[::ds_step]

	return (x_data_ds, y_data_ds)

def remove_samples(sample_label, x_data, y_data):

	rem_indices = np.where(y_data == sample_label)[0]
	
	mask = np.ones(len(y_data), dtype=bool)
	mask[rem_indices] = False

	x_data_ds = x_data[mask, :, :, :]
	y_data_ds = y_data[mask]

	return (x_data_ds, y_data_ds)

def downsample_to_balance(x_data, y_data, num_sample_per_class=None):

	bin_count = np.bincount(y_data)
	
	if num_sample_per_class is None:
		num_sample_per_class = np.min(bin_count)

	print("[balance_samples]: each class has # of samples:", num_sample_per_class)

	mask = np.ones(len(y_data), dtype=bool)

	kept_indices = np.zeros(0).astype(int)
	for label in np.unique(y_data):
		label_indices = np.array(np.where(y_data == label)[0])

		if len(label_indices) <= num_sample_per_class:
			kept_indices = np.concatenate((kept_indices, label_indices))
		else:
			np.random.seed(0)

			np.random.shuffle(label_indices)

			kept_indices = np.concatenate((kept_indices, label_indices[:num_sample_per_class]))

	x_data_ds = x_data[kept_indices, :, :, :]
	y_data_ds = y_data[kept_indices]
	
	return (x_data_ds, y_data_ds)


def upsample_to_balance(x_data, y_data, verbose=False):

	bin_count = np.bincount(y_data)
	num_sample_per_class = np.max(bin_count)

	print("[upsample_to_balance]: each class has # of samples:", num_sample_per_class)

	# x_data_us = np.zeros((0, x_data.shape[1], x_data.shape[2], x_data.shape[3]))
	# y_data_us = np.zeros(0).astype(int)

	x_data_list = []
	y_data_list = []
	for label in np.unique(y_data):
		label_indices = np.array(np.where(y_data == label)[0])
		
		copy_times = num_sample_per_class / len(label_indices)
		
		if verbose:
			print("[upsample_to_balance] class", label, "has", len(label_indices), "samples, and will repeat", copy_times, "times.")

		x_data_label = x_data[label_indices, :, :, :]
		y_data_label = y_data[label_indices]

		x_data_repeat = np.zeros(0)
		y_data_repeat = np.zeros(0)

		if copy_times > 0:
			x_data_repeat = np.repeat(x_data_label, copy_times, axis=0)
			y_data_repeat = np.repeat(y_data_label, copy_times, axis=0)
		else:
			x_data_repeat = x_data_label
			y_data_repeat = y_data_label
		
		x_data_list.append(x_data_repeat)
		y_data_list.append(y_data_repeat)

	x_data_us = np.concatenate(x_data_list)
	y_data_us = np.concatenate(y_data_list)

	return (x_data_us, y_data_us)

def upsample_with_copy(x_data, y_data, labels_to_upsample=None, copy_times=None, gaussian_sigma=None, verbose=False):

	x_data_list = []
	y_data_list = []
	for label, cp_time in zip(labels_to_upsample, copy_times):
		
		label_indices = np.array(np.where(y_data == label)[0])
		
		if verbose:
			print("[upsample_to_balance] class", label, "has", len(label_indices), "samples, and will repeat", cp_time, "times.")

		x_data_label = x_data[label_indices, :, :, :]
		y_data_label = y_data[label_indices]

		x_data_repeat = np.zeros(0)
		y_data_repeat = np.zeros(0)

		if cp_time > 0:
			x_data_repeat = np.repeat(x_data_label, cp_time, axis=0)
			y_data_repeat = np.repeat(y_data_label, cp_time, axis=0)
		else:
			x_data_repeat = x_data_label
			y_data_repeat = y_data_label
		
		if gaussian_sigma is not None:
			x_data_repeat = _add_gaussion(x_data_repeat, gaussian_sigma)

		x_data_list.append(x_data_repeat)
		y_data_list.append(y_data_repeat)

	x_data_us = np.concatenate(x_data_list)
	y_data_us = np.concatenate(y_data_list)

	x_data_us = np.concatenate((x_data, x_data_us))
	y_data_us = np.concatenate((y_data, y_data_us))

	return (x_data_us, y_data_us)

def _add_gaussion(x_data_repeat, gaussian_sigma=1):

	# 1D Gaussian filter

	number_of_points = x_data_repeat.shape[2]/2
	repeat_time = x_data_repeat.shape[1]/2

	x_data = x_data_repeat[:, 0, :number_of_points, :]
	x_data = x_data.reshape([-1, number_of_points, x_data_repeat.shape[3]])

	x_data_r = scipy.ndimage.filters.gaussian_filter1d(x_data[:, :, 0], gaussian_sigma, axis=1)
	x_data_a = scipy.ndimage.filters.gaussian_filter1d(x_data[:, :, 1], gaussian_sigma, axis=1)
	x_data_s = scipy.ndimage.filters.gaussian_filter1d(x_data[:, :, 2], gaussian_sigma, axis=1)

	x_data_gauss = np.dstack((x_data_r, x_data_a, x_data_s))

	feat_1221_2112_repeat_gauss = _feat_to_3D(x_data_gauss, repeat_time)

	# print 'x_data_repeat shape:', x_data_repeat.shape	
	# print 'x_data_gauss shape:', x_data_gauss.shape
	# print 'feat_1221_2112_repeat shape:', feat_1221_2112_repeat_gauss.shape

	return feat_1221_2112_repeat_gauss

def split_data(x_data, y_data, split_rate):
    
	n = x_data.shape[0]

	n_first = int(n * split_rate)

	#np.random.seed(0)
	p_indices = np.random.permutation(n)

	p_indices_first = p_indices[:n_first]
	p_indices_second = p_indices[n_first:]
	
	x_data_first = x_data[p_indices_first, :]
	y_data_first = y_data[p_indices_first]
	
	x_data_second = x_data[p_indices_second, :]
	y_data_second = y_data[p_indices_second]

	return (x_data_first, y_data_first, x_data_second, y_data_second, p_indices_first, p_indices_second)

def bilateralize_feature_OLD(y_names, y_data, x_data, fliped_copy=False):

	all_y_labels = np.sort(np.unique(y_data))
	
	# Replace y label of right structure to its corresponding left structure
	# Note: here we assume that right structure is next to the left structure
	y_data_bilateral_ = y_data
	for y_label in all_y_labels:
		if 'left' in y_names[y_label]:
			# print y_label, y_names[y_label], y_names[y_label+1]
			y_data_bilateral_[np.where(y_data == y_label+1)[0]] = y_label

	# Update y names and re-index y label (to make them continuous numbers)
	all_y_labels = np.sort(np.unique(y_data_bilateral_))
	y_names_bilateral = list()
	y_data_bilateral = y_data_bilateral_
	for idx, y_label in enumerate(all_y_labels):
		y_data_bilateral[np.where(y_data_bilateral_ == y_label)[0]] = idx
		y_names_bilateral.append(y_names[y_label].replace('_left', ''))

	y_names_bilateral = np.array(y_names_bilateral)
	
	# Make bilateral x data
	x_data_reflected = x_data
	x_data_reflected[:,:,:,0] = -x_data_reflected[:,:,:,0]

	x_data_bilateral = np.concatenate((x_data, x_data_reflected), axis=2)

	# augment for training 
	if fliped_copy:
		x_data_bilateral_fliped = np.concatenate((x_data_reflected, x_data), axis=2)
		x_data_bilateral = np.concatenate((x_data_bilateral, x_data_bilateral_fliped), axis=0)
		y_data_bilateral = np.concatenate((y_data_bilateral, y_data_bilateral))
		
	return y_names_bilateral, y_data_bilateral, x_data_bilateral


# def bilateral_feature_name(y_data, y_names):

# 	all_y_labels = np.sort(np.unique(y_data))
	
# 	for y_label in all_y_labels:
# 		if 'left' in y_names[y_label]:
# 			# print y_label, y_names[y_label], y_names[y_label+1]
# 			y_data[np.where(y_data == y_label+1)[0]] = y_label

# 	all_y_labels = np.sort(np.unique(y_data))
# 	y_names_new = list()
# 	for idx, y_label in enumerate(all_y_labels):
# 		y_data[np.where(y_data == y_label)[0]] = idx
# 		y_names_new.append(y_names[y_label])

# 	y_names_new = np.array(y_names_new)
# 	return y_data, y_names_new

def combine_tract_subdiviations_and_keep_outlier_tracts(y_data, y_names, y_validation=None, verbose=False):

	# others
	labels_to_combine = [22, 58, 74, 111, 114, 151] # ['T_FalsePositive' 'T_O_FalsePositive' 'T_O_Partial' 'T_O_Unclassified'  'T_Partial' 'T_Unclassified']

	y_data, y_names = _combine_tracts(y_data, labels_to_combine, combined_name='T_Others', y_names=y_names, verbose=verbose)
	if y_validation is not None:
		y_validation, _ = _combine_tracts(y_validation, labels_to_combine)

	# Superficial tracts: true positive
	labels_to_combine = range(127, 142, 2) # Sup-T left

	y_data, y_names = _combine_tracts(y_data, labels_to_combine, combined_name='T_Sup_left', y_names=y_names, verbose=verbose)
	if y_validation is not None:
		y_validation, _ = _combine_tracts(y_validation, labels_to_combine)

	labels_to_combine = range(128, 143, 2) # Sup-T right

	y_data, y_names = _combine_tracts(y_data, labels_to_combine, combined_name='T_Sup_right', y_names=y_names, verbose=verbose)
	if y_validation is not None:
		y_validation, _ = _combine_tracts(y_validation, labels_to_combine)

	# Superficial tracts: outliers
	labels_to_combine = range(87, 102, 2) # O-Sup-T left

	y_data, y_names = _combine_tracts(y_data, labels_to_combine, combined_name='T_O_Sup_left', y_names=y_names, verbose=verbose)
	if y_validation is not None:
		y_validation, _ = _combine_tracts(y_validation, labels_to_combine)

	labels_to_combine = range(88, 103, 2) # O-Sup-T right

	y_data, y_names = _combine_tracts(y_data, labels_to_combine, combined_name='T_O_Sup_right', y_names=y_names, verbose=verbose)
	if y_validation is not None:
		y_validation, _ = _combine_tracts(y_validation, labels_to_combine)

	# CBLM tracts
	labels_to_combine = [29, 31] # CBLM left

	y_data, y_names = _combine_tracts(y_data, labels_to_combine, combined_name='T_CBLM_left', y_names=y_names, verbose=verbose)
	if y_validation is not None:
		y_validation, _ = _combine_tracts(y_validation, labels_to_combine)

	labels_to_combine = [30, 32] # CBLM right
	y_data, y_names = _combine_tracts(y_data, labels_to_combine, combined_name='T_CBLM_right', y_names=y_names, verbose=verbose)
	if y_validation is not None:
		y_validation, _ = _combine_tracts(y_validation, labels_to_combine)

	labels_to_combine = [65, 67] # O-CBLM left

	y_data, y_names = _combine_tracts(y_data, labels_to_combine, combined_name='T_O_CBLM_left', y_names=y_names, verbose=verbose)
	if y_validation is not None:
		y_validation, _ = _combine_tracts(y_validation, labels_to_combine)

	labels_to_combine = [66, 68] # O-CBLM right
	y_data, y_names = _combine_tracts(y_data, labels_to_combine, combined_name='T_O_CBLM_right', y_names=y_names, verbose=verbose)
	if y_validation is not None:
		y_validation, _ = _combine_tracts(y_validation, labels_to_combine)

	return y_data, y_names, y_validation


def combine_tract_subdiviations_and_merge_outliers(y_data, y_names, y_validation=None, verbose=False):

	# Others
	labels_to_combine = range(36, 112) + [22, 58, 74, 111, 114, 151] # All T_O_* tacts and ['T_FalsePositive' 'T_O_FalsePositive' 'T_O_Partial' 'T_O_Unclassified'  'T_Partial' 'T_Unclassified']

	y_data, y_names = _combine_tracts(y_data, labels_to_combine, combined_name='T_Others', y_names=y_names, verbose=verbose)
	if y_validation is not None:
		y_validation, _ = _combine_tracts(y_validation, labels_to_combine)

	# Superficial tracts: true positive
	labels_to_combine = range(127, 142, 2) # Sup-T left

	y_data, y_names = _combine_tracts(y_data, labels_to_combine, combined_name='T_Sup_left', y_names=y_names, verbose=verbose)
	if y_validation is not None:
		y_validation, _ = _combine_tracts(y_validation, labels_to_combine)

	labels_to_combine = range(128, 143, 2) # Sup-T right

	y_data, y_names = _combine_tracts(y_data, labels_to_combine, combined_name='T_Sup_right', y_names=y_names, verbose=verbose)
	if y_validation is not None:
		y_validation, _ = _combine_tracts(y_validation, labels_to_combine)

	# CBLM tracts: true positive
	labels_to_combine = [29, 31] # CBLM left

	y_data, y_names = _combine_tracts(y_data, labels_to_combine, combined_name='T_CBLM_left', y_names=y_names, verbose=verbose)
	if y_validation is not None:
		y_validation, _ = _combine_tracts(y_validation, labels_to_combine)

	labels_to_combine = [30, 32] # CBLM right
	y_data, y_names = _combine_tracts(y_data, labels_to_combine, combined_name='T_CBLM_right', y_names=y_names, verbose=verbose)
	if y_validation is not None:
		y_validation, _ = _combine_tracts(y_validation, labels_to_combine)


	return y_data, y_names, y_validation

def combine_truepositive_and_falsepositive(y_data, y_names, y_validation=None, verbose=False):

	for y_label, y_name in enumerate(y_names):

		if 'T_O_' in y_name:
			
			tp_tract_name = 'T_'+y_name.replace('T_O_', '')
			idx_tp_tract = np.where(y_names == tp_tract_name)[0].tolist()[0]
			
			labels_to_combine = [y_label, idx_tp_tract]

			y_data, y_names = _combine_tracts(y_data, labels_to_combine, combined_name=y_names[idx_tp_tract], y_names=y_names, verbose=verbose)

			if y_validation is not None:
				y_validation, _ = _combine_tracts(y_validation, labels_to_combine)

	return y_data, y_names, y_validation
		
def bilateralize_feature(y_data, y_names, y_validation=None, verbose=False):

	for y_label, y_name in enumerate(y_names):

		if '_right' in y_name:

			left_tract_name = y_name.replace('_right', '_left')
			idx_left_tract = np.where(y_names == left_tract_name)[0].tolist()[0]
			
			labels_to_combine = [y_label, idx_left_tract]
			combined_name = y_names[idx_left_tract].replace('_left', '')

			y_data, y_names = _combine_tracts(y_data, labels_to_combine, combined_name=combined_name, y_names=y_names, verbose=verbose)

			if y_validation is not None:
				y_validation, _ = _combine_tracts(y_validation, labels_to_combine)

	return y_data, y_names, y_validation

def bilateral_X_data(x_data, fliped_copy=False, y_data=None):

	x_data_reflected = x_data
	x_data_reflected[:,:,:,0] = -x_data_reflected[:,:,:,0]

	if x_data_reflected.shape[-1] == 4:
		print(' # reflect freesurfer labels')
		x_data_reflected[:,:,:,-1] = -x_data_reflected[:,:,:,-1]

		tmp = x_data_reflected[:,:,:,-1]

		c = ma.masked_where(np.absolute(tmp) <= 13, tmp)

		tmp[c.mask] = -tmp[c.mask]
		x_data_reflected[:,:,:,-1] = tmp

	x_data_bilateral = np.concatenate((x_data, x_data_reflected), axis=2)

	if fliped_copy:
		x_data_bilateral_2 = np.concatenate((x_data_reflected, x_data), axis=2)
		x_data_bilateral = np.concatenate((x_data_bilateral, x_data_bilateral_2), axis=0)
		y_data = np.concatenate((y_data, y_data))
	else:
		y_data = None

	return x_data_bilateral, y_data


def _combine_tracts(y_data, labels_to_combine, combined_name=None, y_names=None, verbose=False):

	if verbose:
		if combined_name is not None:
			print('## combine to [', combined_name,']:', y_names[labels_to_combine])

	# all_y_labels = np.sort(np.unique(y_data))
	# for y_label in all_y_labels:
		
	for y_label in labels_to_combine[:-1]:
		y_data[np.where(y_data == y_label)[0]] = labels_to_combine[-1]

	if combined_name is not None:
		y_names[labels_to_combine[-1]] = combined_name
		y_names[labels_to_combine[:-1]] = ''
	else:
		y_names = None

	return y_data, y_names

def compress_labels_and_names(y_train, y_names, y_validation=None):
	
	y_names_new = list()
	y_train_new = y_train
	y_validation_new = y_validation
	
	y_label_new_cc = 0
	for y_label, y_name in enumerate(y_names):
		if y_name != '':
			y_names_new.append(y_name)
			y_train_new[np.where(y_train == y_label)[0]] = y_label_new_cc
			
			if y_validation is not None:
				y_validation_new[np.where(y_validation == y_label)[0]] = y_label_new_cc

			y_label_new_cc = y_label_new_cc + 1
			
	return y_train_new, y_names_new, y_validation_new

def get_tract_specific_data(tract, y_names, y_data, x_data, y_validation=None, x_validation=None, idx_data=None, idx_validation=None):

	kept_labels = list()	
	for y_label, y_name in enumerate(y_names):

		if y_name.endswith(tract) or "_"+tract+"_" in y_name:
			kept_labels.append(y_label)

	if not (len(kept_labels) == 4 or len(kept_labels) == 2):
		print("Error: only two or four labels should be found, but now it has", y_names[kept_labels])
		exit()

	mask = np.ones(len(y_names), dtype=bool)
	mask[kept_labels] = False
	y_names[mask] = ''

	mask = np.zeros(len(y_data), dtype=bool)
	for kept_y_label in kept_labels:
		mask[y_data==kept_y_label] = True

	y_data = y_data[mask]
	x_data = x_data[mask, :]

	if idx_data is not None:
		idx_data = idx_data[mask]

	if y_validation is not None:
		mask = np.zeros(len(y_validation), dtype=bool)
		for kept_y_label in kept_labels:
			mask[y_validation==kept_y_label] = True

		y_validation = y_validation[mask]
		x_validation = x_validation[mask, :]
		idx_validation = idx_validation[mask]

	return y_names, y_data, x_data, y_validation, x_validation, idx_data, idx_validation


def update_y_test_based_on_model_y_names(y_test, y_names, y_names_in_model):

	y_test_updated = y_test.copy()
	y_test_updated[:] = np.nan
	for y_label_in_model, y_name_in_model in enumerate(y_names_in_model):
		
		y_label_in_y_test = np.where(y_names==y_name_in_model)[0]

		if len(y_label_in_y_test) == 0:
			print('Warning: tract [', y_name_in_model, '] does not in the test data.')
			continue

		y_test_updated[y_test == y_label_in_y_test] = y_label_in_model

	return y_test_updated

##

def normalize_channel(x_data, pre_max_v_list=None):

	max_v_list = list()
	for c_idx in range(x_data.shape[-1]):

		print(' # normalizing:', c_idx)
		if pre_max_v_list is None:
			max_v = np.max(x_data[..., c_idx])
		else:
			max_v = pre_max_v_list[c_idx]
		
		max_v_list.append(max_v)

		x_data[..., c_idx] = x_data[..., c_idx] / max_v

	pre_max_v_list = max_v_list

	return x_data, pre_max_v_list





