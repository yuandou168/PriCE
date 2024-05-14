import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Ignore all FutureWarning warnings that might flood the console log
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os
# os.environ["PATH"] = "E:\\Histology\\WSIs\\openslide-win64-20171122\\bin" + ";" + os.environ["PATH"]
# os.environ["PATH"] = "E:\\Histology\\WSIs\\vips-dev-8.11\\bin" + ";" + os.environ["PATH"]
# os.environ["PATH"] = "E:\\Histology\\WSIs\\vips-dev-8.14\\bin" + ";" + os.environ["PATH"]

import pyvips as vips
import openslide
# print("Pyips: ", vips.__version__)
# print("Openslide: ", openslide.__version__)
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from skimage.morphology import closing, opening, dilation, square
from skimage.measure import label, regionprops
from PIL import Image
import json
import pprint
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from torchstat import stat
from torchvision import models
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.patches as mpatches
import timm

import torch.multiprocessing as mp
# mp.get_context('forkserver') # fixes problems with CUDA and fork ('forkserver')
# from multiprocessing import Pool
#from concurrent.futures import ThreadPoolExecutor
# from multiprocess import Pool
# from multiprocessing.pool import ThreadPool as Pool
# from pathos.multiprocessing import ProcessingPool as Pool
# Pool methods which allows tasks to be offloaded to the worker processes in a few different ways.
# mp.get_context('forkserver') # fixes problems with CUDA and fork ('forkserver')

#from thop import profile
#from thop import clever_format
#from deepspeed.profiling.flops_profiler import get_model_profile
#from deepspeed.profiling.flops_profiler import FlopsProfiler
#from pthflops import count_ops
#from flopth import flopth
#from fvcore.nn import FlopCountAnalysis
from numerize import numerize
#from ptflops import get_model_complexity_info

font = {'family': 'serif',
        'weight': 'normal',
        'size': 28}
plt.rc('font', **font)

colors = {"binary": (0.9, 0.9, 0.9),
          "blood": (0.99, 0, 0),
          "damage": (0, 0.5, 0.8),
          "airbubbles": (0, 0.1, 0.5),
          "fold": (0, 0.9, 0.1),
          "blur": (0.99, 0.0, 0.50)}

format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

test_transform = transforms.Compose([
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def sav_fig(path, img, sav_name, cmap="RGB"):
    plt.clf()
    plt.axis("off")
    plt.title(None)
    if cmap == "gray":
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)
    plt.savefig(os.path.join(path, f"{sav_name}.png"), bbox_inches='tight', pad_inches=0)

def remove_small_regions(img, size2remove=100):
    # Closing of the image and gather the labels of the image
    # kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    # img = closing(img, kernel)
    img = closing(img, square(2))
    label_image = label(img)

    # Run through the image labels and set the small regions to zero
    props = regionprops(label_image)
    for region in props:
        if region.area < size2remove:
            minY, minX, maxY, maxX = region.bbox
            img[minY:maxY, minX:maxX] = 0
    return img

def create_binary_mask(wsi_dir, f, sav_path, downsize = 224):
# def create_binary_mask(wsi_dir, f, sav_path, downsize = patch_size):
    # using histolab
    # curr_slide = Slide(os.path.join(location,file),os.path.join(location,file))
    # tissue_mask = curr_slide.scaled_image(100)
    #using openslide
    # slide = openslide.OpenSlide(os.path.join(location,fname))
    # (w,h) = slide.dimensions

    # using pyvips
    print("\n##########################################")
    print(f"Creating basic binary masks for {f}")
    st = time.time()
    file_pth = os.path.join(wsi_dir, f)
    img_400x = read_vips(file_pth)
    w, h = img_400x.width, img_400x.height
    if "#binary#mask.png" not in os.listdir(sav_path):
        thumbnail = img_400x.resize(1/downsize)
        # thumbnail = img_400x.thumbnail_image(round(w/downsize), height=round(h/downsize))
        sav_fig(sav_path, thumbnail, sav_name="#thumbnail")
        # tissue_mask = cv2.cvtColor(np.array(thumbnail), cv2.COLOR_RGBA2RGB)
        img_hsv = cv2.cvtColor(np.array(thumbnail), cv2.COLOR_RGB2HSV)
        mask_HSV = cv2.inRange(img_hsv, (100, 10, 50), (180, 255, 255))
        mask = remove_small_regions(mask_HSV)
        maskInv = cv2.bitwise_not(mask)
        maskInv_closed = remove_small_regions(maskInv)
        binarymask = cv2.bitwise_not(maskInv_closed)
        sav_fig(sav_path, binarymask, sav_name="#binary#mask", cmap='gray')
    print(f"Time taken for creating binary mask {time.time()-st:.2f} seconds")
    return w, h

def fetch(region, patch_size, x, y):
    return region.fetch(patch_size * x, patch_size * y, patch_size, patch_size)

def crop(region, patch_size, x, y):
    return region.crop(patch_size * x, patch_size * y, patch_size, patch_size)

def read_vips(file_path, level=0):
    if file_path.endswith("mrxs"): # mrxs are scanned with
        #  flatten() to force RGBA to RGB, to set a white background
        # print("MRXS file, loading file at 40x")
        try:
            img_400x = vips.Image.new_from_file(file_path, level=level+1,
                                                autocrop=True).flatten()
        except:
            img_400x = vips.Image.new_from_file(file_path, page=level+1,
                                                autocrop=True).flatten()
    else:
        try:
            img_400x = vips.Image.new_from_file(file_path, level=level,
                                                autocrop=True).flatten()
        except:
            img_400x = vips.Image.new_from_file(file_path, page=level,
                                                autocrop=True).flatten()
    return img_400x

def convert_wsi(path,file):
    # The access="sequential" will let libvips stream the image from the source JPEG,
    # rather than decoding the whole thing in advance.
    image = vips.Image.new_from_file(os.path.join(path, file))
    image.write_to_file(f"{file}.tif", tile=True, pyramid=True, compression="jpeg")
    #Pyramidal TIFF images store their levels in one of two ways -- either as pages of the document,
    # or using subifds. A page-based TIFF pyramid stores the pyramid with the full
    # resolution image in page 0, the half-resolution level in page 1, and so on. Use something like:



def create_patches_coordinates(location, file, path, patch_folder, workers=1,
                   patch_size=224, mask_overlap= 95.0):

    # global extract_and_save_patch
    print(f"Create patches for {file}, using {workers} CPUs out of {mp.cpu_count()}")
    # "C:\files" + "/" + "wsi_files"
    datasetname = file.split(".")[0]
    file_pth = os.path.join(location, file)
    # Extract patches from the slide
    st = time.time()
    if file.endswith("mrxs"): # mrxs are scanned with
        #  flatten() to force RGBA to RGB, to set a white background
        # import pyvips as vips
        print("MRXS file, loading file at 40x")
        try:
            img_400x = vips.Image.new_from_file(file_pth, level=1,
                                                autocrop=True).flatten()
        except:
            img_400x = vips.Image.new_from_file(file_pth, page=1,
                                                autocrop=True).flatten()
    else:
        try:
            img_400x = vips.Image.new_from_file(file_pth, level=0,   # 40x
                                                autocrop=True).flatten() #RGBA to RGB img_400x = img_400[:3]
        except:
            img_400x = vips.Image.new_from_file(file_pth, page=0,
                                                autocrop=True).flatten()

    w, h = img_400x.width, img_400x.height
    n_across = int(w/patch_size)
    n_down = int(h/patch_size)
    # n_across * n_down
    list_cord = [(x, y) for x in range(0, n_across) for y in range(0, n_down)]
    mask_path = os.path.join(path, "#binary#mask.png")

    mask_w, mask_h = Image.open(mask_path).size
    resized_mask = vips.Image.new_from_file(mask_path).flatten() \
        .colourspace("b-w").resize(w/mask_w, vscale=h/mask_h)
    img_400x = vips.Region.new(img_400x)
    resized_mask = vips.Region.new(resized_mask)
    # def extract_and_save_patch(x_cord, y_cord, file_path=file_pth, file_name=file, mask_path=mask_path,
                            #    patch_folder=patch_folder, patch_size=patch_size, mask_overlap=mask_overlap):
    coordinates = []
    patches = []
    
    for x_cord, y_cord in list_cord:
        # print(x_cord, y_cord)
        
        # fetch, crop, extract_area,
        patch_mask = fetch(resized_mask, patch_size, x_cord, y_cord)
        patch_mask = np.ndarray(buffer=patch_mask, dtype=np.uint8, shape=[patch_size, patch_size])
        if np.mean(patch_mask/255)*100 > mask_overlap:
            patch = fetch(img_400x, patch_size, x_cord, y_cord)
            patch = np.ndarray(buffer=patch, dtype=np.uint8, shape=[patch_size, patch_size, 3])
            
            # fname = file.split(".")[0]
            x_start, y_start = x_cord*patch_size, y_cord*patch_size
            
            coordinates.append((x_start, y_start))
            # base_name = f"{fname}_{x_start}_{y_start}.png"
            # patch_pil = Image.fromarray(patch)
            # patches.append(patch_pil)
            # patch_pil.save(os.path.join(patch_folder, base_name))
    print("HAHAHA")    
    
    def hash(datasetname,x,y):
        return f'{datasetname},{x},{y}'
    id_to_coordinates = {hash(datasetname, k[0], k[1]): k for k in coordinates}
# print(coordinates, patches, id_to_coordinates)
    time_elapsed = time.time() - st
    minutes = time_elapsed/60
    # print(f"Patches created for {file} in {minutes:.2f} minutes.")
    time_elapsed = time.time() - st
    minutes = time_elapsed/60
    print(f"{len(coordinates)} Patches created for {file} in {minutes:.2f} minutes.")
    
    return coordinates, id_to_coordinates

def create_patches_coordinates_v1(location, file, path, workers=1,
                   patch_size=224, mask_overlap= 95.0):

    global extract_and_save_coornidate
    
    print(f"Create patches for {file}, using {workers} CPUs out of {mp.cpu_count()}")
    # "C:\files" + "/" + "wsi_files"
    file_pth = os.path.join(location, file)
    st = time.time()
    if file.endswith("mrxs"): # mrxs are scanned with
        #  flatten() to force RGBA to RGB, to set a white background
        # import pyvips as vips
        print("MRXS file, loading file at 40x")
        try:
            img_400x = vips.Image.new_from_file(file_pth, level=1,
                                                autocrop=True).flatten()
        except:
            img_400x = vips.Image.new_from_file(file_pth, page=1,
                                                autocrop=True).flatten()

    else:
        try:
            img_400x = vips.Image.new_from_file(file_pth, level=0,   # 40x
                                                autocrop=True).flatten() #RGBA to RGB img_400x = img_400[:3]
        except:
            img_400x = vips.Image.new_from_file(file_pth, page=0,
                                                autocrop=True).flatten()

    w, h = img_400x.width, img_400x.height
    n_across = int(w/patch_size)
    n_down = int(h/patch_size)
    # n_across * n_down
    mask_path = os.path.join(path, "#binary#mask.png")
    
    def extract_and_save_coornidate(x_cord, y_cord, file_path=file_pth, file_name=file, mask_path=mask_path,
                             patch_size=patch_size, mask_overlap=mask_overlap):

        id_to_coordinates = {}
        if file.endswith("mrxs"): # mrxs are scanned with
        #  flatten() to force RGBA to RGB, to set a white background
            print("MRXS file, loading file at 40x")
            try:
                img_400x = vips.Image.new_from_file(file_path, level=1,
                                                    autocrop=True).flatten()
            except:
                img_400x = vips.Image.new_from_file(file_path, page=1,
                                                    autocrop=True).flatten()
        else:
            try:
                img_400x = vips.Image.new_from_file(file_path, level=0,
                                                    autocrop=True).flatten()
            except:
                img_400x = vips.Image.new_from_file(file_path, page=0,
                                                    autocrop=True).flatten()

        w, h = img_400x.width, img_400x.height
        mask_w, mask_h = Image.open(mask_path).size
        resized_mask = vips.Image.new_from_file(mask_path).flatten() \
            .colourspace("b-w").resize(w/mask_w, vscale=h/mask_h)
        img_400x = vips.Region.new(img_400x)
        resized_mask = vips.Region.new(resized_mask)
        # fetch, crop, extract_area,
        fname = file_name.split(".")[0]
        patch_mask = fetch(resized_mask, patch_size, x_cord, y_cord)
        patch_mask = np.ndarray(buffer=patch_mask, dtype=np.uint8, shape=[patch_size, patch_size])
        if np.mean(patch_mask/255)*100 > mask_overlap:
            # patch = fetch(img_400x, patch_size, x_cord, y_cord)
            # patch = np.ndarray(buffer=patch, dtype=np.uint8, shape=[patch_size, patch_size, 3])
            
            x_start, y_start = x_cord*patch_size, y_cord*patch_size
            # base_name = f"{fname}_{x_start}_{y_start}.png"
            # patch_pil = Image.fromarray(patch)
            # patches.append(patch_pil)
            # coordinates.append((x_start, y_start))
            # patch_pil.save(os.path.join(patch_folder, base_name))
        
            def hash(fname,x,y):
                return f'{fname},{x},{y}'
            id_to_coordinates = {hash(fname, x_start, y_start): (x_start, y_start)}
    
        return id_to_coordinates 
    # if workers == 1:
    #     for y in range(0, n_down):
    #         for x in range(0, n_across):
    #             extract_and_save_patch(x, y)
    # else:
    list_cord = [(x, y) for x in range(0, n_across) for y in range(0, n_down)]
    with mp.Pool(processes=workers) as p: # multiprocessing.cpu_count()  # check available CPU counts
        id_to_coordinates = list(p.starmap(extract_and_save_coornidate, list_cord))
    
    id_to_coordinates = [dict(t) for t in set([tuple(d.items()) for d in id_to_coordinates])]
    id_to_coordinates.remove({})
    IDs = {}
    for i in id_to_coordinates:
        # print(i)
        IDs.update(i)

    # print(res)

    time_elapsed = time.time() - st
    minutes = time_elapsed/60
    print(f"Patches coordinates created for {file} in {minutes:.2f} minutes.")
    # print(len(id_to_coordinates), len(IDs.keys()))
    return IDs


def create_patches_v1(location, file, path, patch_folder, workers=1,
                   patch_size=224, mask_overlap= 95.0):

    # global extract_and_save_patch
    print(f"Create patches for {file}, using {workers} CPUs out of {mp.cpu_count()}")
    # "C:\files" + "/" + "wsi_files"
    datasetname = file.split(".")[0]
    file_pth = os.path.join(location, file)
    # Extract patches from the slide
    st = time.time()
    if file.endswith("mrxs"): # mrxs are scanned with
        #  flatten() to force RGBA to RGB, to set a white background
        # import pyvips as vips
        print("MRXS file, loading file at 40x")
        try:
            img_400x = vips.Image.new_from_file(file_pth, level=1,
                                                autocrop=True).flatten()
        except:
            img_400x = vips.Image.new_from_file(file_pth, page=1,
                                                autocrop=True).flatten()
    else:
        try:
            img_400x = vips.Image.new_from_file(file_pth, level=0,   # 40x
                                                autocrop=True).flatten() #RGBA to RGB img_400x = img_400[:3]
        except:
            img_400x = vips.Image.new_from_file(file_pth, page=0,
                                                autocrop=True).flatten()

    w, h = img_400x.width, img_400x.height
    n_across = int(w/patch_size)
    n_down = int(h/patch_size)
    # n_across * n_down
    list_cord = [(x, y) for x in range(0, n_across) for y in range(0, n_down)]
    mask_path = os.path.join(path, "#binary#mask.png")

    mask_w, mask_h = Image.open(mask_path).size
    resized_mask = vips.Image.new_from_file(mask_path).flatten() \
        .colourspace("b-w").resize(w/mask_w, vscale=h/mask_h)
    img_400x = vips.Region.new(img_400x)
    resized_mask = vips.Region.new(resized_mask)
    # def extract_and_save_patch(x_cord, y_cord, file_path=file_pth, file_name=file, mask_path=mask_path,
                            #    patch_folder=patch_folder, patch_size=patch_size, mask_overlap=mask_overlap):
    coordinates = []
    patches = []
    
    for x_cord, y_cord in list_cord:
        # print(x_cord, y_cord)
        
        # fetch, crop, extract_area,
        patch_mask = fetch(resized_mask, patch_size, x_cord, y_cord)
        patch_mask = np.ndarray(buffer=patch_mask, dtype=np.uint8, shape=[patch_size, patch_size])
        if np.mean(patch_mask/255)*100 > mask_overlap:
            patch = fetch(img_400x, patch_size, x_cord, y_cord)
            patch = np.ndarray(buffer=patch, dtype=np.uint8, shape=[patch_size, patch_size, 3])
            
            # fname = file.split(".")[0]
            x_start, y_start = x_cord*patch_size, y_cord*patch_size
            
            coordinates.append((x_start, y_start))
            # base_name = f"{fname}_{x_start}_{y_start}.png"
            patch_pil = Image.fromarray(patch)
            patches.append(patch_pil)
            # patch_pil.save(os.path.join(patch_folder, base_name))
    print("HAHAHA")    
    
    def hash(datasetname,x,y):
        return f'{datasetname},{x},{y}'
    id_to_coordinates = {hash(datasetname, k[0], k[1]): k for k in coordinates}
# print(coordinates, patches, id_to_coordinates)
    time_elapsed = time.time() - st
    minutes = time_elapsed/60
    # print(f"Patches created for {file} in {minutes:.2f} minutes.")
    time_elapsed = time.time() - st
    minutes = time_elapsed/60
    print(f"Patches created for {file} in {minutes:.2f} minutes.")
    
    return coordinates, id_to_coordinates, patches

   
def create_patches_v3(location, file, path, patch_folder, workers=1,
                   patch_size=224, mask_overlap= 95.0):

    global extract_and_save_patch
    coordinates = []
    patches = []

    print(f"Create patches for {file}, using {workers} CPUs out of {mp.cpu_count()}")
    # "C:\files" + "/" + "wsi_files"
    file_pth = os.path.join(location, file)
    st = time.time()
    if file.endswith("mrxs"): # mrxs are scanned with
        #  flatten() to force RGBA to RGB, to set a white background
        # import pyvips as vips
        print("MRXS file, loading file at 40x")
        try:
            img_400x = vips.Image.new_from_file(file_pth, level=1,
                                                autocrop=True).flatten()
        except:
            img_400x = vips.Image.new_from_file(file_pth, page=1,
                                                autocrop=True).flatten()

    else:
        try:
            img_400x = vips.Image.new_from_file(file_pth, level=0,   # 40x
                                                autocrop=True).flatten() #RGBA to RGB img_400x = img_400[:3]
        except:
            img_400x = vips.Image.new_from_file(file_pth, page=0,
                                                autocrop=True).flatten()

    w, h = img_400x.width, img_400x.height
    n_across = int(w/patch_size)
    n_down = int(h/patch_size)
    # n_across * n_down
    mask_path = os.path.join(path, "#binary#mask.png")
    datasetname = file.split(".")[0]
    def extract_and_save_patch(x_cord, y_cord, file_path=file_pth, file_name=file, mask_path=mask_path,
                               patch_folder=patch_folder, patch_size=patch_size, mask_overlap=mask_overlap):
        
        if file.endswith("mrxs"): # mrxs are scanned with
        #  flatten() to force RGBA to RGB, to set a white background
            print("MRXS file, loading file at 40x")
            try:
                img_400x = vips.Image.new_from_file(file_path, level=1,
                                                    autocrop=True).flatten()
            except:
                img_400x = vips.Image.new_from_file(file_path, page=1,
                                                    autocrop=True).flatten()
        else:
            try:
                img_400x = vips.Image.new_from_file(file_path, level=0,
                                                    autocrop=True).flatten()
            except:
                img_400x = vips.Image.new_from_file(file_path, page=0,
                                                    autocrop=True).flatten()

        w, h = img_400x.width, img_400x.height
        mask_w, mask_h = Image.open(mask_path).size
        resized_mask = vips.Image.new_from_file(mask_path).flatten() \
            .colourspace("b-w").resize(w/mask_w, vscale=h/mask_h)
        img_400x = vips.Region.new(img_400x)
        resized_mask = vips.Region.new(resized_mask)
        # fetch, crop, extract_area,
        patch_mask = fetch(resized_mask, patch_size, x_cord, y_cord)
        patch_mask = np.ndarray(buffer=patch_mask, dtype=np.uint8, shape=[patch_size, patch_size])
        if np.mean(patch_mask/255)*100 > mask_overlap:
            patch = fetch(img_400x, patch_size, x_cord, y_cord)
            patch = np.ndarray(buffer=patch, dtype=np.uint8, shape=[patch_size, patch_size, 3])
            fname = file_name.split(".")[0]
            x_start, y_start = x_cord*patch_size, y_cord*patch_size
            base_name = f"{fname}_{x_start}_{y_start}.png"
            patch_pil = Image.fromarray(patch)
            patches.append(patch_pil)
            coordinates.append((x_start, y_start))
            patch_pil.save(os.path.join(patch_folder, base_name))
        
        
        def hash(datasetname,x,y):
            return f'{datasetname},{x},{y}'
        id_to_coordinates = {hash(datasetname, k[0], k[1]): k for k in coordinates}
        
        # print(len(coordinates), len(patches))
        return coordinates, id_to_coordinates, patches 
    # if workers == 1:
    #     for y in range(0, n_down):
    #         for x in range(0, n_across):
    #             extract_and_save_patch(x, y)
    # else:
    list_cord = [(x, y) for x in range(0, n_across) for y in range(0, n_down)]
    with mp.Pool(processes=workers) as p: # multiprocessing.cpu_count()  # check available CPU counts
        res = p.starmap(extract_and_save_patch, list_cord)
        # args = [(file, img_400x, resized_mask, patch_folder,
        #          cord_tuple, patch_size, mask_overlap) for cord_tuple in list_cord]
        # if bool(res[0]) == True: 
    # print(res)
    final = []
    for i in res:
        if i != ([], {}, []):
            time.sleep(3)
            print(i)
            final.append(i)
    # final = list(set(final))
    time_elapsed = time.time() - st
    minutes = time_elapsed/60
    print(f"Patches created for {file} in {minutes:.2f} minutes.")
    print(len(final))
    # return tile_counter
    # return coordinates, id_to_coordinates, patches 


# def extract_and_save_patch(y_cord, file_path, file_name, mask_path,
#                            patch_folder, patch_size=224, mask_overlap=95.0):
#     slide =  read_vips(file_path)
#     mask = vips.Image.new_from_file(mask_path)
#     resized_mask = mask.resize(slide.width/mask.width,
#                                vscale=slide.height/mask.height, kernel="nearest")
#     n_across = int(slide.width/ patch_size)
#     for x_cord in range(n_across):
#         patch_mask = crop(resized_mask, patch_size, x_cord, y_cord)
#         if patch_mask.avg()/2.55 > mask_overlap:
#             # print(f"average overlap of patch {patch_mask.avg()/2.55}")
#             patch = crop(slide, patch_size, x_cord, y_cord)

#             fname = file_name.split(".")[0]
#             x_start, y_start = x_cord*patch_size, y_cord*patch_size
#             base_name = f"{fname}_{x_start}_{y_start}.png"
#             patch.write_to_file(os.path.join(patch_folder, base_name))
    
def create_patches_v2(location, file, path, patch_folder, workers=1,
                   patch_size=224, mask_overlap= 95.0):

    global extract_and_save_patch
    
    print(f"Create patches for {file}, using {workers} CPUs out of {mp.cpu_count()}")
    # "C:\files" + "/" + "wsi_files"
    file_pth = os.path.join(location, file)
    st = time.time()
    if file.endswith("mrxs"): # mrxs are scanned with
        #  flatten() to force RGBA to RGB, to set a white background
        # import pyvips as vips
        print("MRXS file, loading file at 40x")
        try:
            img_400x = vips.Image.new_from_file(file_pth, level=1,
                                                autocrop=True).flatten()
        except:
            img_400x = vips.Image.new_from_file(file_pth, page=1,
                                                autocrop=True).flatten()

    else:
        try:
            img_400x = vips.Image.new_from_file(file_pth, level=0,   # 40x
                                                autocrop=True).flatten() #RGBA to RGB img_400x = img_400[:3]
        except:
            img_400x = vips.Image.new_from_file(file_pth, page=0,
                                                autocrop=True).flatten()

    w, h = img_400x.width, img_400x.height
    n_across = int(w/patch_size)
    n_down = int(h/patch_size)
    # n_across * n_down
    mask_path = os.path.join(path, "#binary#mask.png")
    
    def extract_and_save_patch(x_cord, y_cord, file_path=file_pth, file_name=file, mask_path=mask_path,
                               patch_folder=patch_folder, patch_size=patch_size, mask_overlap=mask_overlap):
        
        patches = []
        id_to_patch = {}
        if file.endswith("mrxs"): # mrxs are scanned with
        #  flatten() to force RGBA to RGB, to set a white background
            print("MRXS file, loading file at 40x")
            try:
                img_400x = vips.Image.new_from_file(file_path, level=1,
                                                    autocrop=True).flatten()
            except:
                img_400x = vips.Image.new_from_file(file_path, page=1,
                                                    autocrop=True).flatten()
        else:
            try:
                img_400x = vips.Image.new_from_file(file_path, level=0,
                                                    autocrop=True).flatten()
            except:
                img_400x = vips.Image.new_from_file(file_path, page=0,
                                                    autocrop=True).flatten()

        w, h = img_400x.width, img_400x.height
        mask_w, mask_h = Image.open(mask_path).size
        resized_mask = vips.Image.new_from_file(mask_path).flatten() \
            .colourspace("b-w").resize(w/mask_w, vscale=h/mask_h)
        img_400x = vips.Region.new(img_400x)
        resized_mask = vips.Region.new(resized_mask)
        # fetch, crop, extract_area,
        patch_mask = fetch(resized_mask, patch_size, x_cord, y_cord)
        patch_mask = np.ndarray(buffer=patch_mask, dtype=np.uint8, shape=[patch_size, patch_size])
        if np.mean(patch_mask/255)*100 > mask_overlap:
            patch = fetch(img_400x, patch_size, x_cord, y_cord)
            patch = np.ndarray(buffer=patch, dtype=np.uint8, shape=[patch_size, patch_size, 3])
            fname = file_name.split(".")[0]
            x_start, y_start = x_cord*patch_size, y_cord*patch_size
            base_name = f"{fname}_{x_start}_{y_start}.png"
            patch_pil = Image.fromarray(patch)
            patch_pil.save(os.path.join(patch_folder, base_name))
            # if patch_pil != []:
            patches.append(patch)

            # def hash(fname,x,y):
            #     return f'{fname},{x},{y}'
            # id_to_patch = {hash(fname, x_start, y_start): patch_pil}
    
        return patches

    # if workers == 1:
    #     for y in range(0, n_down):
    #         for x in range(0, n_across):
    #             extract_and_save_patch(x, y)
    # else:

    list_cord = [(x, y) for x in range(0, n_across) for y in range(0, n_down)]
    with mp.Pool(processes=workers) as p: # multiprocessing.cpu_count()  # check available CPU counts
        patches = p.starmap(extract_and_save_patch, list_cord)
        # args = [(file, img_400x, resized_mask, patch_folder,
        #          cord_tuple, patch_size, mask_overlap) for cord_tuple in list_cord]
    
    # print(len(patches))
    # id_to_patch = [dict(t) for t in set([tuple(d.items()) for d in id_to_patch])]
    # import itertools
    # patches.sort()
    # list(patches for patches,_ in itertools.groupby(patches))
    # patches.remove([])
    # IDs = {}
    # for i in id_to_patch:
        # print(i)
        # IDs.update(i)

    # print(res)
    # print(patches, len(patches))
    time_elapsed = time.time() - st
    minutes = time_elapsed/60
    print(f"Patches created for {file} in {minutes:.2f} minutes.")
    # print(len(id_to_patch), len(IDs.keys()))
    return patches
    


def create_patches(location, file, path, patch_folder,
                   workers=1, patch_size=224, mask_overlap=95.0):

    print(f"Create patches for {file}, using {workers} CPU out of {mp.cpu_count()}")
    file_path = os.path.join(location, file)
    st = time.time()

    img_400x = read_vips(file_path)
    w, h = img_400x.width, img_400x.height
    n_across = int(w/patch_size)
    n_down = int(h/patch_size)
    # n_across * n_down
    list_cord = [(x, y) for x in range(0, n_across) for y in range(0, n_down)]
    
    mask_path = os.path.join(path, "#binary#mask.png")

    params = [((x, y), file_path, file, mask_path, patch_folder, patch_size, mask_overlap)
            for x in range(0, n_across) for y in range(0, n_down)]

    with mp.Pool(processes=workers) as p:

        result = p.starmap(extract_and_save_patch, params)
        # result = p.starmap(extract_patches_coords, params)

    time_elapsed = time.time() - st
    minutes = time_elapsed/60
    print(f"{n_down} Patches created for {file} in {minutes:.2f} minutes.")


def data_generator(patch_folder, test_transform, batch_size=32, worker=1):
    print(f"\nLoading patches...........")
    # test_images = datasets.ImageFolder(root=patch_folder, transform= test_transform)
    test_images = custom_data_loader(patch_folder, test_transform)
    test_loader = DataLoader(dataset=test_images, batch_size=batch_size, shuffle=False, num_workers=worker, pin_memory=True)
    total_patches = len(test_images)
    print(f"total number of patches are {total_patches}")
    return test_loader, total_patches

def load_cnn_model(weight_loc, weights_name, num_classes=2, dropout=0.2):
    model = models.mobilenet_v3_large()
    model.classifier = custom_classifier(960, num_classes, dropout=dropout)
    best_model_wts = os.path.join(weight_loc, weights_name)
    model.load_state_dict(torch.load(best_model_wts, map_location=torch.device('cpu'))['model'])
    # if model_name is not None:
    # flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224),), verbose=False)
    # flops, params = clever_format([flops, params], "%.3f")
    # print(f"{model_name}: Thops flops {flops},    Params {params}\n")
        # stat(model, (3,224,224))
        # flops, macs, params = get_model_profile(model=model, input_shape = (1,3,224,224),
        #                                         detailed=False, print_profile=False)
        # print(f"{model_name}:  {flops} FLOPs,  {macs}, {params} Params\n")

        # flops = FlopCountAnalysis(model, torch.rand(1, 3, 224, 224))
        # print(f"{model_name}: FLOPs {flops.total()}\n")
    model.eval()
    return model

def load_vit_model(weight_loc, weights_name, num_classes=2):
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_classes)
    best_model_wts = os.path.join(weight_loc, weights_name)
    model.load_state_dict(torch.load(best_model_wts, map_location=torch.device('cpu'))['model'])
    # if model_name is not None:
    #     flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224),), verbose=False)
    #     flops, params = clever_format([flops, params], "%.3f")
    #     print(f"{model_name}: Thops flops {flops}, Params {params}\n")
    #     #
    #     flops1, params = flopth(model)
    #     print(f"{model_name}: Flopth flops {flops1},   Params {params}\n")
    #     stat(model, (3, 224, 224))
    #     flops, macs, params = get_model_profile(model, (1, 3, 224, 224),
    #                                             detailed=False, print_profile=False)
    #     prof = FlopsProfiler(model)
    #     prof.start_profile()
    #     flops = prof.get_total_flops()
    #     macs = prof.get_total_macs()
    #     params = prof.get_total_params()
    #     prof.end_profile()
    #     # flops, flops_dic = count_ops(model, torch.rand(1, 3, 224, 224), print_readable=False, verbose=False)
    #     flops = FlopCountAnalysis(model, torch.rand(1, 3, 224, 224))
    #     print(f"{model_name}: FLOPs {numerize.numerize(flops.total())}\n")
    model.eval()
    return model

    # inputs = torch.randn(40, 16, 18, 260)
    # with profiler.profile(record_shapes=True, with_flops=True) as prof:
    #     model(inputs)
    #     profiler_output = prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10)
    #     print(profiler_output)

class custom_data_loader(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_dir = img_path
        self.transform = transform
        self.data_path = []
        file_list = os.listdir(self.img_dir)
        for img in file_list:
            self.data_path.append(os.path.join(self.img_dir, img))

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        image = Image.open(self.data_path[idx])
        label = 0
        if self.transform is not None:
            return self.transform(image), label
        else:
            return image, label

class custom_classifier(nn.Module):
    def __init__(self, in_features, num_classes, dropout=0.2):
        super(custom_classifier, self).__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x)) # fully connected layer 1
        x = self.dropout(x)
        feat = F.relu(self.fc2(x)) # fully connected layer 2
        x = self.dropout(x)
        x = self.fc3(feat)   #fully connected layer 3
        return x, feat

def infer_multiclass(model, test_loader, use_prob_threshold = None):
    y_preds, probs, artifact_free, blood, blur, bubble, damage, fold = [], [], [], [], [], [], [], []
    for data, target in test_loader:
        if torch.cuda.is_available():
            data = data.cuda()
            # ID to classes  {0: 'artifact_free', 1: 'blood', 2: 'blur',
            # 3: 'bubble', 4: 'damage', 5: 'fold'}
        try:
            output, _ = model(data)
        except:
            output = model(data)

        probabilities = F.softmax(output, dim=1)
        if use_prob_threshold is not None:
            preds = (probabilities >= use_prob_threshold)
            _, preds = torch.max(preds, 1)
        else:
            _, preds = torch.max(output, 1)
        # probabilities = F.softmax(output, dim=1).detach().cpu().numpy()
        probs.append(list(np.around(probabilities.detach().cpu().numpy(), decimals=5)))
        y_pred = preds.cpu().numpy()

        artifact_free.append(list((y_pred == 0).astype(int)))
        blood.append(list((y_pred == 1).astype(int)))
        blur.append(list((y_pred == 2).astype(int)))
        bubble.append(list((y_pred == 3).astype(int)))
        damage.append(list((y_pred == 4).astype(int)))
        fold.append(list((y_pred == 5).astype(int)))

        y_preds.append(list(y_pred))

    artifact_free = convert_batch_list(artifact_free)
    blood = convert_batch_list(blood)
    blur = convert_batch_list(blur)
    bubble = convert_batch_list(bubble)
    damage = convert_batch_list(damage)
    fold = convert_batch_list(fold)
    probs = convert_batch_list(probs)
    y_preds = convert_batch_list(y_preds)

    return y_preds, artifact_free, blood,  blur, bubble, damage, fold, probs

def infer_cnn(model, test_loader, use_prob_threshold = None):
    y_pred, probs = [], []
    for data, target in test_loader:
        if torch.cuda.is_available():
            data = data.cuda()
        output, _ = model(data)
        probabilities = F.softmax(output, dim=1)
        if use_prob_threshold is not None:
            preds = (probabilities >= use_prob_threshold)
            _, preds = torch.max(preds, 1)
        else:
            _, preds = torch.max(output, 1)
        probs.append(list(np.around(probabilities.detach().cpu().numpy(), decimals=5)))
        y_pred.append(list(preds.cpu().numpy()))
    return convert_batch_list(y_pred), convert_batch_list(probs)

def infer_vit(model, test_loader, use_prob_threshold = None):
    y_preds, probs = [], []
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = model(data)
            probabilities = F.softmax(output, dim=1)
            if use_prob_threshold is not None:
                preds = (probabilities >= use_prob_threshold)
                _, preds = torch.max(preds, 1)
            else:
                _, preds = torch.max(output, 1)
            # probabilities = F.softmax(output, dim=1).detach().cpu().numpy()
            probs.append(list(np.around(probabilities.detach().cpu().numpy(), decimals=5)))
            y_preds.append(list(preds.cpu().numpy()))
        return convert_batch_list(y_preds),  convert_batch_list(probs)

def convert_batch_list(lst_of_lst):
    return sum(lst_of_lst, [])

def post_process_masks(dataf, mask_saving_path, wsi_shape, downsize=224, blur=True, blood=True,
                       damage=True, fold=True, airbubble=True, merged=True ):
    # dataframe can be loaded from excel sheet instead
    # dataf = pd.read_excel(path_to_excel, engine='openpyxl')
    if blood:
        print("-----Producing masks for blood-------------")
        blood_df = dataf[dataf['blood'] == 1]
        mask_shape = (round(wsi_shape[1]/downsize), round(wsi_shape[0]/downsize))# h,w
        blood_mask = np.full(mask_shape, False)
        for name in blood_df['files'].to_list():
            # for patch naming style SUShud37_30_8064_6048.png
            x_cord = int(name.split(".")[0].split("_")[-2])
            y_cord = int(name.split(".")[0].split("_")[-1])
            blood_mask[int(y_cord/downsize), int(x_cord/downsize)] = True
        sav_fig(mask_saving_path, Image.fromarray(blood_mask).convert("L"), sav_name="#blood#mask", cmap='gray')
    if blur:
        print("-----Producing masks for blur-------------")
        blur_df = dataf[dataf['blur'] == 1]
        mask_shape = (round(wsi_shape[1]/downsize), round(wsi_shape[0]/downsize))# h,w
        blur_mask = np.full(mask_shape, False)
        for name in blur_df['files'].to_list():
            # for patch naming style SUShud37_30_8064_6048.png
            x_cord = int(name.split(".")[0].split("_")[-2])
            y_cord = int(name.split(".")[0].split("_")[-1])
            blur_mask[int(y_cord/downsize), int(x_cord/downsize)] = True
        sav_fig(mask_saving_path, Image.fromarray(blur_mask).convert("L"), sav_name="#blur#mask", cmap='gray')
    if damage:
        print("-----Producing masks for damaged tissue--")
        damage_df = dataf[dataf['damage'] == 1]
        mask_shape = (round(wsi_shape[1]/downsize), round(wsi_shape[0]/downsize))# h,w
        damage_mask = np.full(mask_shape, False)
        for name in damage_df['files'].to_list():
            # for patch naming style SUShud37_30_8064_6048.png
            x_cord = int(name.split(".")[0].split("_")[-2])
            y_cord = int(name.split(".")[0].split("_")[-1])
            damage_mask[int(y_cord/downsize), int(x_cord/downsize)] = True
        sav_fig(mask_saving_path, Image.fromarray(damage_mask).convert("L"), sav_name="#damage#mask", cmap='gray')
    if fold:
        print("-----Producing masks for folded tissue---")
        fold_df = dataf[dataf['fold'] == 1]
        mask_shape = (round(wsi_shape[1]/downsize), round(wsi_shape[0]/downsize))# h,w
        fold_mask = np.full(mask_shape, False)
        for name in fold_df['files'].to_list():
            # for patch naming style SUShud37_30_8064_6048.png
            x_cord = int(name.split(".")[0].split("_")[-2])
            y_cord = int(name.split(".")[0].split("_")[-1])
            fold_mask[int(y_cord/downsize), int(x_cord/downsize)] = True
        sav_fig(mask_saving_path, Image.fromarray(fold_mask).convert("L"), sav_name="#fold#mask", cmap='gray')
    if airbubble:
        print("-----Producing masks for airbubbles-------")
        airbubble_df = dataf[dataf['airbubble'] == 1]
        mask_shape = (round(wsi_shape[1]/downsize), round(wsi_shape[0]/downsize))# h,w
        airbubbles_mask = np.full(mask_shape, False)
        for name in airbubble_df['files'].to_list():
            # for patch naming style SUShud37_30_8064_6048.png
            x_cord = int(name.split(".")[0].split("_")[-2])
            y_cord = int(name.split(".")[0].split("_")[-1])
            airbubbles_mask[int(y_cord/downsize), int(x_cord/downsize)] = True
        sav_fig(mask_saving_path, Image.fromarray(airbubbles_mask).convert("L"),
                sav_name="#airbubbles#mask", cmap='gray')
    if merged:
        print("-----Producing masks a merged mask--------")
        merge_masks(mask_saving_path)
def merge_masks(path):
    listofmasks = os.listdir(path)
    listofmasks = [f for f in listofmasks if f.endswith("png") and not (f.startswith("#binary") or f.startswith("#merged")
                                        or f.startswith("#artifact") or f.startswith("#thumb") or f.startswith("#segmentation"))]
    shape = Image.open(os.path.join(path, listofmasks[0])).size
    output_mask = np.full((shape[1], shape[0]), False)
    for img in listofmasks:
        mask = Image.open(os.path.join(path, img)).convert("L")
        mask = mask.resize(shape)
        output_mask = output_mask | np.asarray(mask)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    # dilation(output_mask, kernel)
    Image.fromarray(dilation(output_mask, kernel)).save(f"{path}/#merged#mask.png", quality=95)
    # sav_fig(path, Image.fromarray(output_mask).convert("L"), sav_name="merged_mask", cmap='gray')

def artifact_free_mask(path):
    binary_mask = Image.open(os.path.join(path, "#binary#mask.png")).convert("L")
    artifact_mask = Image.open(os.path.join(path, "#merged#mask.png")).convert("L")
    shape = artifact_mask.size
    binary_mask = np.asarray(binary_mask.resize(shape), dtype=np.bool)
    artifact_mask = np.asarray(artifact_mask, dtype=np.bool)
    output_mask = binary_mask.astype(int) - artifact_mask.astype(int)
    output_mask = (output_mask == 1)
    # kernel = np.ones((2, 2))
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    final_mask = Image.fromarray(opening(output_mask, kernel)).convert("L")
    sav_fig(path, final_mask, sav_name="#artifactfree#mask", cmap='gray')

def segmentation_color_mask(path, colors= colors, scale=1):
    listofmasks = os.listdir(path)
    listofmasks = [f for f in listofmasks if f.endswith("png") and not (f.startswith("#merged") or
                        f.startswith("#segm") or f.startswith("#artifactfree") or f.startswith("#thumb"))]
    shape = Image.open(os.path.join(path, listofmasks[0])).size # Take shape of One_mask
    seg_img = np.zeros(shape=(shape[1], shape[0], 3))# make a 3D array
    legend_patch = []
    # first mask color comes first on the colormap
    # sort to make binary mask come first
    sorted_list = sorted(listofmasks, key=lambda x: x.startswith("#b"), reverse=True)
    for f in sorted_list:
        mask_type = f.split("#")[1]
        mask = Image.open(os.path.join(path, f)).convert("L")
        mask_1 = np.asarray(mask.resize(shape), dtype=np.float32)
        seg_img[:, :, 0] += (mask_1 * (colors[mask_type][0]))
        seg_img[:, :, 1] += (mask_1 * (colors[mask_type][1]))
        seg_img[:, :, 2] += (mask_1 * (colors[mask_type][2]))
        if mask_type == "binary":
            legend_patch.append(mpatches.Patch(color=colors[mask_type], label="Tissue"))
        else:
            legend_patch.append(mpatches.Patch(color=colors[mask_type], label=mask_type.capitalize()))

    plt.clf()
    plt.axis("off")
    plt.title(None)
    plt.legend(handles=legend_patch, loc='best', fontsize=10, framealpha=0.3, facecolor="y")

    if scale > 1:
        im = Image.fromarray((seg_img*255).astype(np.uint8), 'RGB')
        im = im.resize((im.width*scale, im.height*scale))
        # plt.legend(handles=legend_patch, loc='best', fontsize= 12, framealpha = 0.3, facecolor="y")
        plt.imshow(im)
        plt.savefig(f"{path}/#segmentation#mask.png", dpi=300, bbox_inches='tight', pad_inches=0)
        # resized_im.save(f"{path}/segmentation_mask.png", quality=95, bbox_inches='tight', pad_inches=0) # does not give legend
        # sav_fig(path, resized_im, sav_name="segmentation_mask")
    else:
        im = Image.fromarray((seg_img*255).astype(np.uint8), 'RGB')
        # plt.imshow((seg_img*255).astype(np.uint8))
        #sav_fig(path, Image.fromarray((seg_img*255).astype(np.uint8)), sav_name="segmentation_mask")
        # plt.savefig(f"{path}/segmentation_mask.png", dpi=300, bbox_inches='tight', pad_inches=0)
        plt.imshow(im)
        plt.savefig(f"{path}/#segmentation#mask.png", dpi=300, bbox_inches='tight', pad_inches=0)

def refine_artifacts_wsi(path_to_wsi, path):
    artifact_free_mask(path)
    artifactfree_mask = os.path.join(path, "#artifactfree#mask.png")
    mask = vips.Image.new_from_file(artifactfree_mask).flatten()
    if path_to_wsi.endswith("mrxs"): # mrxs are scanned with
        print("File is MRXS")
        wsi = vips.Image.new_from_file(path_to_wsi, level=1, autocrop=True).flatten()
    else:
        wsi = vips.Image.new_from_file(path_to_wsi, level=0, autocrop=True).flatten()
    mask = mask.resize(wsi.width / mask.width, vscale=(wsi.height / mask.height), kernel="nearest")
    wsi *= mask / 255.0
    wsi.write_to_file(f"{path}/refined.tiff",  tile=True, properties=True, tile_width=512,
                      tile_height=512, compression="jpeg", pyramid=True) ### CHECK THIS

def calculate_quality(path_to_masks):
    report = dict()
    listofmasks = os.listdir(path_to_masks)
    listofmasks = [f for f in listofmasks if f.endswith("png") and not
                                            (f.startswith("#merged") or f.startswith("#segm")
                                            or f.startswith("#thumb") or f.startswith("#binar"))]
    baseline_matrix = np.asarray(Image.open(os.path.join(path_to_masks, "#binary#mask.png")).convert("L"), dtype=np.bool)
    total_pixels = np.sum(baseline_matrix == 1)
    for img in listofmasks:
        mask_sum = np.sum(np.asarray(Image.open(os.path.join(path_to_masks, img)).convert("L"), dtype=np.bool) == 1)
        label = img.split("#")[1]
        report[label] = str(round(mask_sum/total_pixels * 100, 2)) + " %"
    pp = pprint.PrettyPrinter(width=41, compact=True)
    pp.pprint(report)
    with open(f"{path_to_masks}/quality_report.json", "w") as f:
        json.dump(report, f, indent=4)

def check_tissue_region(patch):
    patch = np.asarray(patch.getdata())[:, 0]
    val = np.histogram(patch, bins=[100, 235, 255])[0]
    if val[0] < val[1]:
        return False
    else:
        return True


def extract_patches_coords(location, file, path, sav_patch_folder,
                           patch_size, use_mask_to_threshold=True,
                           level=0, overlap=0,
                           threshold=80.0, sav_patches=False, workers=1):
    """
    Extracts patch cords from a whole slide image using pyvips.

    Parameters:
    slide_path (str): The path to the whole slide image.
    patch_size (int): The size of the patches to extract.
    level (int): The level of the slide to extract patches from. Default is 0.
    overlap (int): The overlap between adjacent patches. Default is 0.
    threshold (int): The threshold for patch selection. Default is 0.
    return_coordinates (bool): If True, return the spatial position of each patch. Default is False.

    Returns:
        coordinates (list): A list of tuples containing the spatial position of each patch
    """
    print(f"Create patches for {file}, using {workers} CPU out of {mp.cpu_count()}")
    
    # Load the slide with pyvips
    file_path = os.path.join(location, file)
    st = time.time()

    slide = read_vips(file_path, level=level)
    width, height = slide.width, slide.height

    n_down = int(height/patch_size)
    mask_path = os.path.join(path, "#binary#mask.png")
    params = [(y, file_path, file, mask_path, sav_patch_folder, patch_size, overlap)
              for y in range(0, n_down)]
    
    mask = vips.Image.new_from_file(mask_path)
    resized_mask = mask.resize(width/mask.width,
                               vscale=height/mask.height, kernel="nearest")

    # Compute the number of patches in each direction
    num_patches_x = (width - patch_size) // (patch_size - overlap) + 1
    num_patches_y = (height - patch_size) // (patch_size - overlap) + 1

    # Extract patches from the slide
    coordinates = []
    patches = []
    for i in range(num_patches_x):
        for j in range(num_patches_y):
            x = i * (patch_size - overlap)
            y = j * (patch_size - overlap)
            if use_mask_to_threshold:
                mask_patch = resized_mask.crop(x, y, patch_size, patch_size)
                if mask_patch.avg()/2.55 > threshold:
                    coordinates.append((x, y))
                    patch = slide.crop(x, y, patch_size, patch_size)
                    patches.append(patch)
                    if sav_patches:
                        patch = slide.crop(x, y, patch_size, patch_size)
                        base_name = f"{x}_{y}.png"
                        patch.write_to_file(os.path.join(sav_patch_folder, base_name))

            else:
            # Check if the patch meets the threshold criteria
                patch = slide.crop(x, y, patch_size, patch_size)
                if patch.avg()/2.55 > threshold:
                    coordinates.append((x, y))
                    patch = slide.crop(x, y, patch_size, patch_size)
                    patches.append(patch)
                    if sav_patches:
                        base_name = f"{x}_{y}.png"
                        patch.write_to_file(os.path.join(sav_patch_folder, base_name))

    time_elapsed = time.time() - st
    minutes = time_elapsed/60
    print(f"Total patches {len(coordinates)} created for {file} in {minutes:.2f} minutes.")

    def hash(x,y):
        return f'{x},{y}'

    id_to_coordinates = {hash(k[0], k[1]): k for k in coordinates }

    return coordinates, patches, id_to_coordinates

class WSI_Patch_Dataset(Dataset):
    def __init__(self, slide_path, coords_list, patch_size=224, transform=None):
        self.slide_path = slide_path
        self.coords_list = coords_list
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.coords_list)

    def __getitem__(self, idx):
        # Load the slide image using pyvips
        # slide_image = vips.Image.new_from_file(self.slide_path, access='sequential')
        slide_image = vips.Image.new_from_file(self.slide_path, level=0,
                                            autocrop=True).flatten()

        # Get the coordinates and extract the patch
        x, y = self.coords_list[idx]
        patch = slide_image.extract_area(x, y, self.patch_size, self.patch_size).write_to_memory()

        # Convert the patch to a tensor and apply the transformation
        if self.transform:
            patch = self.transform(patch)
        return patch



def create_foreground_mask_vips(wsi_dir, f, save_path=None, downsize=1):
    # Open the slide image using PyVips
    print("\n##########################################")
    print(f"Creating basic binary masks for {f}")
    st = time.time()
    slide_path = os.path.join(wsi_dir, f)
    slide = read_vips(slide_path)
    if "#binary#mask.png" not in os.listdir(save_path):
    # Downsize the image if requested
        if downsize != 1:
            # slide = slide.reduce(downsize, downsize)
            slide = slide.resize(1/downsize)
        # Convert the image to grayscale and threshold it to create a binary mask
        sav_fig(save_path, slide, sav_name="#thumbnail")
        gray = slide.colourspace('b-w')
        _, binary_mask = cv2.threshold(np.ndarray(buffer=gray.write_to_memory(), dtype=np.uint8, shape=[gray.height, gray.width]),
                                       0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # binary_mask = gray.more(threshold=0, direction='above')

        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = ~binary_mask
        # Save the binary mask if requested
        if save_path is not None:
            # sav_path = os.path.join(save_path, "#binary#mask")
            # cv2.imwrite(sav_path, binary_mask)
            sav_fig(save_path, binary_mask, sav_name="#binary#mask", cmap='gray')

    print(f"Time taken for creating binary mask {time.time()-st:.2f} seconds")
    return slide.width, slide.height

#
# create_foreground_mask_vips("E:\\wsi_data", "CZ464.TP.I.I-3.ndpi", downsize=224, save_path="E:\\wsi_data")
# create_binary_mask("E:\\wsi_data", "CZ464.TP.I.I-3.ndpi", "E:\\wsi_data", downsize=224)