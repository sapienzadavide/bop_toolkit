import os
import time
import argparse
import subprocess
import numpy as np
import sys
import json

import cv2
import glob
import configparser
import csv
from matplotlib import pyplot as plt
from PIL import Image

from bop_toolkit_lib import config
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc

import copy
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import pose_error
from bop_toolkit_lib import renderer
from matplotlib import pyplot as plt



def main(args, p):
    result_filename = args.result_filename
    gt_filetype = args.gt_filetype
    misc.log('===========')
    misc.log('EVALUATING: {}'.format(result_filename))
    misc.log('===========')

    result_name = os.path.splitext(os.path.basename(result_filename))[0]

    # Parse info about the method and the dataset from the filename.
    result_info = result_name.split('_')
    method = str(result_info[0])
    dataset_info = result_info[1].split('-')
    dataset = str(dataset_info[0])
    split = str(dataset_info[1])
    split_type = str(dataset_info[2]) if len(dataset_info) > 2 else None
    split_type_str = ' - ' + split_type if split_type is not None else ''

    # Load dataset parameters.
    '''WARNING: modify dataset params obj_ids in custom, if you want more objects. For now n_objs = 4'''
    dp_split = dataset_params.get_split_params(
        p['datasets_path'], dataset, split, split_type)

    model_type = 'eval'
    dp_model = dataset_params.get_model_params(
        p['datasets_path'], dataset, model_type)

    for error in p['errors']:
        error_type = error['type']

        # Load object models.
        models = {}
        if error_type in ['ad', 'add', 'adi', 'mssd', 'mspd', 'proj']:
            misc.log('Loading object models...')
            for obj_id in dp_model['obj_ids']:
                models[obj_id] = inout.load_ply(
                    dp_model['model_tpath'].format(obj_id=obj_id))

        # Load models info.
        models_info = None
        if error_type in ['ad', 'add', 'adi', 'mssd', 'mspd', 'cus']:
            models_info = inout.load_json(
                dp_model['models_info_path'], keys_to_int=True)

        # Get sets of symmetry transformations for the object models.
        models_sym = None
        if error_type in ['mssd', 'mspd']:
            models_sym = {}
            for obj_id in dp_model['obj_ids']:
                models_sym[obj_id] = misc.get_symmetry_transformations(
                    models_info[obj_id], p['max_sym_disc_step'])

        # Load the estimation targets.
        targets = inout.load_json(
            os.path.join(dp_split['base_path'], p['targets_filename']))

        # Organize the targets by scene, image and object.
        misc.log('Organizing estimation targets...')
        targets_org = {}
        for target in targets:
            targets_org.setdefault(target['scene_id'], {}).setdefault(
                target['im_id'], {})[target['obj_id']] = target

        # Images Filenames
        misc.log('Loading images filenames...')
        file_path = args.file_path
        if os.path.isdir(file_path):
            files = sorted(
                glob.glob(os.path.join(str(file_path), '*.png')) + glob.glob(os.path.join(str(file_path), '*.jpg')))
        else:
            files = [file_path]

        # Load the estimated poses from csv file.
        misc.log('Loading estimated poses...')
        ests = inout.load_bop_results(result_filename)

        # Organize estimated poses
        misc.log('Organizing estimated poses ...')
        ests_org = {}
        for est in ests:
            ests_org.setdefault(est['scene_id'], {}).setdefault(
                est['im_id'], {}).setdefault(est['obj_id'], []).append(est)


        # Load also ground truth IF is a csv file
        if gt_filetype=='csv':
            gt_filename = args.gt_filename
            gt_poses = inout.load_bop_results(gt_filename)

            # Organize estimated poses
            misc.log('Organizing gt poses ...')
            gt_org = {}
            for poses in gt_poses:
                gt_org.setdefault(poses['scene_id'], {}).setdefault(
                    poses['im_id'], {}).setdefault(poses['obj_id'], []).append(poses)

            if gt_filetype=='csv':
                # Insert also scene_camera for now.
                # scene_camera = inout.load_scene_camera(
                #     dp_split['scene_camera_tpath'].format(scene_id=scene_id))

                scene_gt_tot={}
                for k1 in gt_org.keys():
                    scene_gt_tot[k1]={}
                    for k2 in gt_org[k1].keys():
                        for k3 in gt_org[k1][k2].keys():
                            im_id = -1
                            tmp_list = []
                            for i,pred in enumerate(gt_org[k1][k2][k3]):
                                # Check im_ids on all obj id predictions
                                if im_id==-1:
                                    im_id=pred['im_id']
                                else:
                                    assert(im_id==pred['im_id'])
                                tmp_list.append({'cam_R_m2c': gt_org[k1][k2][k3][i]['R'],
                                                 'cam_t_m2c': gt_org[k1][k2][k3][i]['t'],
                                                 'obj_bb': [0, 0, 0, 0],
                                                 'obj_id':gt_org[k1][k2][k3][i]['obj_id']})
                            scene_gt_tot[k1][im_id] = tmp_list

        eval_calc_errors={}
        for scene_id, scene_targets in targets_org.items():

            # Load camera and GT poses for the current scene.
            if gt_filetype=='json':
                # scene_camera = inout.load_scene_camera(
                #     dp_split['scene_camera_tpath'].format(scene_id=scene_id))
                scene_gt = inout.load_scene_gt(dp_split['scene_gt_tpath'].format(
                    scene_id=scene_id))
            elif gt_filetype=='csv':
                scene_gt = scene_gt_tot[scene_id]


            scene_errs = []
            for im_ind, (im_id, im_targets) in enumerate(scene_targets.items()):

                # Intrinsic camera matrix.
                # K = scene_camera[im_id]['cam_K']

                for obj_id, target in im_targets.items():

                    # The required number of top estimated poses.
                    if p['n_top'] == 0:  # All estimates are considered.
                        n_top_curr = None
                    elif p['n_top'] == -1:  # Given by the number of GT poses.
                        n_top_curr = target['inst_count']
                    else:
                        n_top_curr = p['n_top']

                    # Get the estimates.
                    try:
                        obj_ests = ests_org[scene_id][im_id][obj_id]
                    except KeyError:
                        obj_ests = []

                    # Sort the estimates by score (in descending order).
                    obj_ests_sorted = sorted(
                        enumerate(obj_ests), key=lambda x: x[1]['score'], reverse=True)

                    # Select the required number of top estimated poses.
                    obj_ests_sorted = obj_ests_sorted[slice(0, n_top_curr)]

                    # Calculate error of each pose estimate w.r.t. all GT poses of the same
                    # object class.
                    for est_id, est in obj_ests_sorted:
                        # Estimated pose.
                        R_e = est['R']
                        t_e = est['t']
                        # Compute errors
                        errs = {}  # Errors w.r.t. GT poses of the same object class.
                        for gt_id, gt in enumerate(scene_gt[im_id]):
                            if gt['obj_id'] != obj_id:
                                continue
                            # Ground-truth pose.
                            R_g = gt['cam_R_m2c']
                            t_g = gt['cam_t_m2c']
                            # Check if the projections of the bounding spheres of the object in
                            # the two poses overlap (to speed up calculation of some errors).
                            sphere_projections_overlap = None
                            if error['type'] in ['cus']:
                                radius = 0.5 * models_info[obj_id]['diameter']
                                sphere_projections_overlap = misc.overlapping_sphere_projections(
                                    radius, t_e.squeeze(), t_g.squeeze())

                            # Check if the bounding spheres of the object in the two poses
                            # overlap (to speed up calculation of some errors).
                            spheres_overlap = None
                            if error['type'] in ['ad', 'add', 'adi', 'mssd']:
                                center_dist = np.linalg.norm(t_e - t_g)
                                spheres_overlap = center_dist < models_info[obj_id]['diameter']

                            if error['type'] == 'mssd':
                                if not spheres_overlap:
                                    e = [float('inf')]
                                else:
                                    e = [pose_error.mssd(
                                        R_e, t_e, R_g, t_g, models[obj_id]['pts'],
                                        models_sym[obj_id])]

                            elif error['type'] == 'mspd':
                                e = [pose_error.mspd(
                                    R_e, t_e, R_g, t_g, K, models[obj_id]['pts'],
                                    models_sym[obj_id])]

                            elif error['type'] in ['ad', 'add', 'adi']:
                                if not spheres_overlap:
                                    # Infinite error if the bounding spheres do not overlap. With
                                    # typically used values of the correctness threshold for the AD
                                    # error (e.g. k*diameter, where k = 0.1), such pose estimates
                                    # would be considered incorrect anyway.
                                    e = [float('inf')]
                                else:
                                    if error['type'] == 'ad':
                                        if obj_id in dp_model['symmetric_obj_ids']:
                                            e = [pose_error.adi(
                                                R_e, t_e, R_g, t_g, models[obj_id]['pts'])]
                                        else:
                                            e = [pose_error.add(
                                                R_e, t_e, R_g, t_g, models[obj_id]['pts'])]

                                    elif error['type'] == 'add':
                                        e = [pose_error.add(
                                            R_e, t_e, R_g, t_g, models[obj_id]['pts'])]

                                    else:  # 'adi'
                                        e = [pose_error.adi(
                                            R_e, t_e, R_g, t_g, models[obj_id]['pts'])]

                            elif error['type'] == 'cus':
                                if sphere_projections_overlap:
                                    e = [pose_error.cus(
                                        R_e, t_e, R_g, t_g, K, ren, obj_id)]
                                else:
                                    e = [1.0]

                            elif error['type'] == 'proj':
                                e = [pose_error.proj(
                                    R_e, t_e, R_g, t_g, K, models[obj_id]['pts'])]

                            elif error['type'] == 'rete':
                                e = [pose_error.re(R_e, R_g), pose_error.te(t_e, t_g)]

                            elif error['type'] == 're':
                                e = [pose_error.re(R_e, R_g)]

                            elif error['type'] == 'te':
                                e = [pose_error.te(t_e, t_g)]

                            else:
                                raise ValueError('Unknown pose error function.')

                            errs[gt_id] = e
                        # Save the calculated errors.
                        scene_errs.append({
                            'im_id': im_id,
                            'obj_id': obj_id,
                            'est_id': est_id,
                            'score': est['score'],
                            'errors': errs
                        })
            eval_calc_errors[scene_id] = scene_errs
        misc.log('Errors have been computed!')
        misc.log('=====================================================')

        # Accuracy computation phase.
        misc.log('Given errors, compute the recall based on ADD(-S) error, with {} as method and {} as dataset'.format
                 (method, dataset))
        tars = 0
        tp = 0
        for scene_id, scene_errs in eval_calc_errors.items():
            threshold = error['correct_th']
            th = np.float64(float(threshold) * models_info[int(scene_id)]['diameter'])
            tars += len(scene_errs)
            tp_obj = 0
            tars_obj = len(scene_errs)
            for x in scene_errs:
                if int(scene_id) == x['obj_id']:
                    if x['errors'][0][0] <= th:
                        tp += 1
                        tp_obj += 1
            misc.log('For object {}, the recall obtained is {}% .'.format(scene_id, (tp_obj / tars_obj)*100))
        print('Final recall:', tp / tars)


if __name__ == '__main__':
    # PARAMETERS (some can be overwritten by the command line arguments below).
    ################################################################################
    p = {
        # Errors to calculate.
        'errors': [
            # {
            #   'n_top': -1,
            #   'type': 'mssd',
            #   'correct_th': [[th] for th in np.arange(0.05, 0.51, 0.05)]
            # },
            # {
            #   'n_top': -1,
            #   'type': 'mspd',
            #   'correct_th': 10
            # },
            # {
            #   'n_top': -1,
            #   'type': 'add',
            #   'correct_th': 0.1
            # },
            {
                'n_top': -1,
                'type': 'ad',
                'correct_th': 0.1
            },

        ],

        # Minimum visible surface fraction of a valid GT pose.
        # -1 == k most visible GT poses will be considered, where k is given by
        # the "inst_count" item loaded from "targets_filename".
        'visib_gt_min': -1,

        # See misc.get_symmetry_transformations().
        'max_sym_disc_step': 0.01,

        # Type of the renderer (used for the VSD pose error function).
        'renderer_type': 'vispy',  # Options: 'vispy', 'cpp', 'python'.

        # Names of files with results for which to calculate the errors (assumed to be
        # stored in folder p['results_path']). See docs/bop_challenge_2019.md for a
        # description of the format. Example results can be found at:
        # http://ptak.felk.cvut.cz/6DB/public/bop_sample_results/bop_challenge_2019/
        'result_filename':
            '/relative/path/to/csv/with/results',

        # Folder with results to be evaluated.
        'results_path': '/relative/path/to/csv_folder/with/results',

        # Folder for the calculated pose errors and performance scores.
        'eval_path': '/relative/path/where/saving/errors',

        # File with a list of estimation targets to consider. The file is assumed to
        # be stored in the dataset folder.
        'targets_filename': 'test_targets_bop19.json',

        # Folder containing the BOP datasets (only for object information and ply models)
        'datasets_path': '/home/elena/repos/datasetMetriche2/',

        # Template of path to the output file with calculated errors.
        'out_errors_tpath': os.path.join(
            '{eval_path}', '{result_name}', '{error_sign}',
            'errors_{scene_id:06d}.json'),

        # Ground-truth pose yml file.
        'gt_filename': '/path/to/yml/GT/file'

    }
    ################################################################################

    # Command line arguments.
    # ------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--renderer_type', default=p['renderer_type'])
    # parser.add_argument('--n_top', default=-1)
    parser.add_argument('--result_filename', default=p['result_filename'],help='files with results.')
    parser.add_argument('--results_path', help='Necessary only if it contains scene_gt and scene_camera', default=p['results_path'])
    parser.add_argument('--eval_path', default=p['eval_path'])
    parser.add_argument('--targets_filename', default=p['targets_filename'])
    parser.add_argument('--gt_filename', help='Necessary only if it is csv', default=p['gt_filename'])
    parser.add_argument("--file_path", help='folder to image(s)',
                        default='/path/to/images_folder/rgb')
    parser.add_argument("-test_config", type=str, required=False, default='test_config_webcam.cfg')
    parser.add_argument("-save_res", type=str, required=False, default='')
    parser.add_argument("-vis", action='store_true', default=False)
    parser.add_argument("-debugvis", action='store_true', default=False)
    parser.add_argument('--gt_filetype', help='file could be .csv or .json format', default='csv' )
    args = parser.parse_args()

    p['renderer_type'] = str(args.renderer_type)
    p['result_filename'] = str(args.result_filename)
    p['results_path'] = str(args.results_path)
#    p['eval_path'] = str(args.eval_path)
    p['targets_filename'] = str(args.targets_filename)
    p['n_top'] = -1

    args = parser.parse_args()
    main(args, p)

