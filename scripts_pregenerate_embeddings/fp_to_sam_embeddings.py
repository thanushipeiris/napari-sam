# this script takes command line argument of a folder and that generates
# SAM embeddings for all IMARIS images contained in that folder.
# It can be submitted using the corresponding shell script as
# qsub fp_to_sam_embeddings.sh -v fp={} \; 

###############################################
############# CHANGE THESE VALUES #############
MODEL = "vit_h"
MODEL_PATH = r'INSERT SAM MODEL FILEPATH/sam_vit_h_4b8939.pth'
embedding_save_fp = r'INSERT FILEPATH WHERE YOU WANT EMBEDDINGS TO BE SAVED/SAM_embeddings'
RESOLUTION_LEVEL = 1 # resolution level of imaris images to be read in
INCLUDE_SUBFOLDERS=True # will iteratively find all imaris images
verbose=True
MAX_Z_SLICE = 100 # reduce this if your computer still runs into a memory error when annotating with napari-sam and regenerate embeddings - it will save embeddings in smaller z stacks
CHANNELS_TO_INPUT_TO_SAM = [1,2,3] # SAM will only handle 3 channels so specify the channels you want to extract from the image, e.g. this is 2nd-4th channels
###############################################
###############################################

import h5py
import numpy as np

def read_ims_to_numpy_ZYXC(fp, resolutionLevel, verbose=False):

    # read ims in as numpy array
    if fp.endswith(".ims"):
        image = get_ZYXC_numpy_from_ims(fp,
                                resolutionLevel=resolutionLevel,
                                verbose=verbose)
    else:
        return Warning("fp is not of filetype .ims")
    if verbose: print("  read image:", image.shape)

    return image

# function that reads in ims filepath and returns np array
def get_ZYXC_numpy_from_ims(fp, resolutionLevel, verbose=False):
    """
    reads in ims filepath and returns np array ZYXC
    @param resolutionLevel: resolution level to be extracted from ims file (assumes it exists, does not check)
    @return: np.array of image
    """
    if verbose: print("reading:", fp)
    with h5py.File(fp, 'r') as hf:
        #print(hf)
        #hf.visit(print) #prints out all nodes in hdf5
        dataset = hf.get("DataSet").get(f"ResolutionLevel {resolutionLevel}").get("TimePoint 0")

        # extract the Data, you also have Histogram btw
        img_ch_list = []
        for channel in dataset.keys():
            #print(channel)
            new_ch = dataset.get(channel).get("Data")
            new_ch = np.array(new_ch)
            img_ch_list.append(new_ch)

        img = np.ascontiguousarray(np.stack(img_ch_list, axis=-1))
        if verbose: print("  size of nparr:", img.shape)
        return img#.astype(np.uint8)


# script that will load imaris files, choose 3 channels and output embedding split into max z stack size of 100

# read in images
import torch
import os
from napari.layers.utils.layer_utils import calc_data_range
from segment_anything import SamPredictor, build_sam_vit_h, build_sam_vit_l, build_sam_vit_b, sam_model_registry
from tqdm import tqdm
import numpy as np

#from napari_sam.utils import normalize
# copied this direct from napari_sam as was not able to import on the hpc
def normalize(x, source_limits=None, target_limits=None):
    if source_limits is None:
        source_limits = (x.min(), x.max())

    if target_limits is None:
        target_limits = (0, 1)

    if source_limits[0] == source_limits[1] or target_limits[0] == target_limits[1]:
        return x * 0
    else:
        x_std = (x - source_limits[0]) / (source_limits[1] - source_limits[0])
        x_scaled = x_std * (target_limits[1] - target_limits[0]) + target_limits[0]
        return x_scaled

# use command line args
import sys
IMS_FILEPATH = str(sys.argv[1])
print(IMS_FILEPATH)

# setup SAM
sam = sam_model_registry[MODEL](checkpoint=MODEL_PATH)
predictor = SamPredictor(sam)

# iteratively find all ims files
if INCLUDE_SUBFOLDERS:
    fps = [os.path.join(path,file) for path,dirc,files in os.walk(IMS_FILEPATH) for file in files if file.endswith(".ims")]
else:
    fps = [f for f in os.scandir(IMS_FILEPATH) if f.endswith(".ims")]
if verbose:
    print(f"{len(fps)} ims files located.")

sum_larger_max_z_slices = 0

# generate and save SAM embeddings for each ims in sets with max no. z slices 100
for fp in sorted(fps):##testing first image
    image = read_ims_to_numpy_ZYXC(fp, RESOLUTION_LEVEL, verbose=verbose)
    image_name = os.path.splitext(os.path.basename(fp))[0]
    print(f"IMAGE NAME {image_name}, z slices {image.shape[0]}")
    if image.shape[0] > MAX_Z_SLICE:
        sum_larger_max_z_slices += 1

    # select channels and autocontrast all channels together
    if image.shape[-1] != 3:
        image = image[...,CHANNELS_TO_INPUT_TO_SAM] # chose channels 1,2,3 (skip first channel 0)
    contrast_limits = calc_data_range(image, rgb=True) # use napari autocontrast to set contrast limits
    print("  contrast limits", contrast_limits)
    
    image_embedding_slices = []
    n_z_stack=0
    
    for i in tqdm(range(image.shape[0]), desc=f"Creating SAM image embedding for {image_name}"):
        #contrast_limits = self.image_layer.contrast_limits
        image_slice = image[i,...]
        
        image_slice = normalize(image_slice, source_limits=contrast_limits, target_limits=(0, 255)).astype(np.uint8)
        predictor.set_image(image_slice)
        embedding = predictor.features
        image_embedding_slices.append(embedding)

        if not (predictor.original_size==predictor.input_size):
            print("WARNING*** PREDICTOR ARGS", predictor.original_size, predictor.input_size, predictor.is_image_set)

        # save stack
        last_z_in_stack = (n_z_stack+1)*MAX_Z_SLICE-1
        if i==last_z_in_stack:
            torch.save(image_embedding_slices, os.path.join(embedding_save_fp, f"{image_name}_zmax{MAX_Z_SLICE}-zstack{n_z_stack}.pt"))
            n_z_stack += 1
            image_embedding_slices = []
            
    torch.cuda.empty_cache()
        
                  
print(f"{sum_larger_max_z_slices} images with z slices > {MAX_Z_SLICE}")
