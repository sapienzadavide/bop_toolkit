'''
This script converts the yolo-6d prediction file format to that required by BOT toolkit.

yolo-6d prediction file: predictions_linemod_*.mat
desired BOT toolkit file: custom-test.csv

In predictions_linemod_*.mat the variables/fields are:
 - corner_gts (nx9x2)
 - corner_prs (nx9x2)
 - R_gts (nx3x3)
 - R_prs (nx3x3)
 - t_gts (nx3)
 - t_prs (nx3)

In custom-test.csv contains only the predictions. Each row contains:
 - scene_id (== obj_id) (int)
 - im_id (int): incremental number associuated to the images
 - obj_id (int)
 - score (float): confidence
 - R (9xfloat)
 - t (3xfloat)
 - time (float)
'''

import numpy as np
import os
import argparse
import configparser
from scipy.io import loadmat
import csv

# only the hotstab is used in this example
OBJ_ID = 1
SCORE = 0.99
TIME = 0.01

def load_mat_file(input_file):
    data = loadmat(input_file)
    return data

def load_testset(test_path):
    f = open(test_path, 'r')
    lines = f.readlines()
    f.close()
    return lines

def get_image_ids(images):
    image_ids = []
    for im in images:
        image_ids.append(int(im.split('/')[-1].split('.')[0]))
    return image_ids

def main(args):
    # check input file 
    if not os.path.exists(args.input_path):
        print("Error: {} does not exist".format(args.input_path))
        exit(-1)
    if args.input_path.split('.')[-1] != 'mat':
        print("Error: {} has the wrong format (.mat is nedeed)".format(args.input_path))
        exit(-1)

    # check test set file 
    if not os.path.exists(args.test_path):
        print("Error: {} does not exist".format(args.test_path))
        exit(-1)
    if args.test_path.split('.')[-1] != 'txt':
        print("Error: {} has the wrong format (.txt is nedeed)".format(args.test_path))
        exit(-1)
    
    # check output file
    if os.path.exists(args.output_path):
        print("WARNING: {} already exists".format(args.output_path))
    if args.output_path.split('.')[-1] != 'csv':
        print("Error: {} has the wrong format (.csv is nedeed)".format(args.output_path))
        exit(-1)
    
    if args.mode != 'pred' and args.mode != 'gt':
        print("Error: mode is {}, but only 'pred' and 'gt' are supported.".format(args.mode))
        exit()

    data = load_mat_file(args.input_path)
    images = load_testset(args.test_path)
    image_ids = get_image_ids(images)

    
    # dict_keys(['__header__', '__version__', '__globals__', 'R_gts', 't_gts', 'corner_gts', 'R_prs', 't_prs', 'corner_prs'])
    assert (data['R_gts'].shape == data['R_prs'].shape )
    assert (data['t_gts'].shape == data['t_prs'].shape )
    assert (data['R_gts'].shape[0] == len(images) )
    assert (data['t_gts'].shape[0] == len(images) )
    assert (len(image_ids) == len(images) )
    
    n_iter = len(images)
    #----------------
    results = {}
    results['scene_id'] = [OBJ_ID for n in range(n_iter)]
    results['im_id'] = image_ids
    results['obj_id'] = [OBJ_ID for n in range(n_iter)]
    results['score'] = [SCORE for n in range(n_iter)]
    if args.mode == 'pred':
        results['R'] = data['R_prs']
        results['t'] = data['t_prs']
    else:
        results['R'] = data['R_gts']
        results['t'] = data['t_gts']    
    results['time'] = [TIME for n in range(n_iter)]
    
    f = open(args.output_path,'w')
    writer = csv.writer(f)
    writer.writerow(results.keys())
    for i in range(len(results['scene_id'])):
        R = np.array(results['R'][i]).flatten()
        t = np.array(results['t'][i]).flatten()
        R_str = ''
        for ri in R:
            R_str = R_str+str(ri)+' '
        t_str = ''
        for ti in t:
            t_str = t_str+str(ti)+' '
        writer.writerow([results['scene_id'][i], results['im_id'][i], results['obj_id'][i], results['score'][i], 
                        R_str, t_str, results['time'][i]])
        
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True, help='input yolo-6d prediction file', default='predictions_linemod_hotstab.mat')
    parser.add_argument("-t", "--test_path", required=True, help='test.txt used in yolo-6d', default='test.txt')
    parser.add_argument("-o", "--output_path", required=True, help='output csv file in BOT toolkit format', default='yolo6d_custom-test.csv')
    parser.add_argument("-m", "--mode", required=True, help='execution mode: on estimations or ground-truths', default='')
    args = parser.parse_args()
    main(args)

