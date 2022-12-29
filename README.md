# BOP Toolkit

A Python toolkit of the BOP benchmark for 6D object pose estimation
(http://bop.felk.cvut.cz).

- **bop_toolkit_lib** - The core Python library for i/o operations, calculation
  of pose errors, Python based rendering etc.
- **docs** - Documentation and conventions.
- **scripts** - Scripts for evaluation, rendering of training images,
  visualization of 6D object poses etc.

## Installation

### Python Dependencies

- Anaconda 4.10.3

To create conda environment: 
```
conda env create -f boptoolkit_env.yml
```

To install the required python libraries, run:
```
conda activate boptoolkit
pip install -r requirements.txt -e .
```

In the case of problems, try to first run: 
```
conda activate boptoolkit 
pip install --upgrade pip setuptools
```

### Vispy Renderer (default)

The Python based headless renderer with egl backend is implemented using [Vispy](https://github.com/vispy/vispy).
Vispy is installed using the pip command above.
Note that the [nvidia opengl driver](https://developer.nvidia.com/opengl-driver) might be required in case of any errors.

### Python Renderer (deprecated)

Another Python based renderer is implemented using
[Glumpy](https://glumpy.github.io/) which depends on
[freetype](https://www.freetype.org/) and [GLFW](https://www.glfw.org/).
This implementation is similar to the vispy renderer since glumpy and vispy have similar apis,
but this renderer does not support headless rendering.
Glumpy is installed using the pip command above. On Linux, freetype and GLFW can
be installed by:

```
apt-get install freetype
apt-get install libglfw3
```

To install freetype and GLFW on Windows, follow [these instructions](https://glumpy.readthedocs.io/en/latest/installation.html#step-by-step-install-for-x64-bit-windows-7-8-and-10).

GLFW serves as a backend of Glumpy. [Another backend](https://glumpy.readthedocs.io/en/latest/api/app-backends.html)
can be used but were not tested with our code.

### C++ Renderer

For fast CPU-based rendering on a headless server, we recommend installing [bop_renderer](https://github.com/thodan/bop_renderer),
an off-screen C++ renderer with Python bindings.

## Usage

### 1. Get the BOP datasets

Download the BOP datasets and make sure they are in the [expected folder structure](https://bop.felk.cvut.cz/datasets/).

### 2. Run your method

Estimate poses and save them in one .csv file per dataset ([format description](https://bop.felk.cvut.cz/challenges/bop-challenge-2020/#howtoparticipate)).

### 3. Configure the BOP Toolkit

In [bop_toolkit_lib/config.py](https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/config.py), set paths to the BOP datasets, to a folder with results to be evaluated, and to a folder for the evaluation output. The other parameters are necessary only if you want to visualize results or run the C++ Renderer.

### 4. Evaluate the pose estimates
```
conda activate boptoolkit 
python scripts/eval_bop19.py --renderer_type=vispy --result_filenames=NAME_OF_CSV_WITH_RESULTS
```
`--renderer_type`: "vispy", "python", or "cpp" (We recommend using "vispy" since it is easy to install and works headlessly. For "cpp", you need to install the C++ Renderer [bop_renderer](https://github.com/thodan/bop_renderer).).

`--result_filenames`: Comma-separated filenames with pose estimates in .csv ([examples](http://ptak.felk.cvut.cz/6DB/public/bop_sample_results)).

### 5. Evaluate the detections / instance segmentations
```
conda activate boptoolkit 
python scripts/eval_bop_coco.py --result_filenames=NAME_OF_JSON_WITH_COCO_RESULTS --ann_type='bbox'
```
--result_filenames: Comma-separated filenames with per-dataset coco results (place them under your `results_path` defined in your [config.py](bop_toolkit_lib/config.py)).  
--ann_type: 'bbox' to evaluate amodal bounding boxes. 'segm' to evaluate segmentation masks.

## Convert BOP to COCO format

```
conda activate boptoolkit 
python scripts/calc_gt_coco.py
```

Set the dataset and split parameters in the top section of the script.

## Manual annotation tool

To annotate a new dataset in BOP format use [this tool](./scripts/annotation_tool.py).

First install Open3d dependency

```
conda activate boptoolkit 
pip install open3d==0.15.2
```

Edit the file paths in parameters section at the beginning of the file then run:

```
conda activate boptoolkit 
python scripts/annotation_tool.py
```

### Interface:

Control the object pose with the following keys
`i`: up, `,`: down, `j`: front, `k`:back, `h`:left, `l`:right

Translation/rotation mode:
- Shift not clicked: translation mode
- Shift clicked: rotation model

Distance/angle big or small:
- Ctrl not clicked: small distance(1mm) / angle(2deg)
- Ctrl clicked: big distance(5cm) / angle(90deg)

R or "Refine" button will call ICP algorithm to do local refinement of the annotation

## Custom eval 

Alternative evaluation script to compute ADD(-S) and Recall.

```
python scripts/est_pose_file2add_accuracy.py --result_filename <path-to-csv-estimation-file> --dataset_path <path-to-dataset-folder> --gt_filename <path-to-groundtruth-file> --gt_filetype <type> 
#python scripts/est_pose_file2add_accuracy.py --result_filename /home/elena/repos/datasetMetriche2/underwater/prova_underwater-test.csv --dataset_path /home/elena/repos/bop_toolkit/yolo6D_evaluation_files --gt_filename /home/elena/repos/datasetMetriche2/underwater/prova_underwater-test.csv
```
where:

 - path-to-csv-estimation-file: path to the csv file with the estimated poses. Each row must contain: scene_id, im_id, obj_id, score, R, t, time.
 - path-to-dataset-folder: path to the dataset folder. It is a folder with the dataset name, containing the test_target_BOP.json file and two folders (models and models_eval, which are the same of BOP dataset)
 - path-to-groundtruth-file: path to the csv or json file with the ground-truth poses. The csv file must be written in the same format of the < path-to-csv-estimation-file > file, while json file must be scene_gt of BOP Dataset format. 
 - type: "json" or "csv" type. It specifies the < path-to-groundtruth-file > file extension.


### Yolo-6D

To evaluate the yolo-6D model please referred to the corresponding repo and README:

```
python valid.py <data-file> <cfg-file> <weights>
# python valid.py custom_cfg/hotstab.data custom_cfg/yolo-pose.cfg backup/hotstab/model.weights
```
Where: 
 - data-file is the required file for training yolo-6D. It contains the trainable objects' names.
 - cfg-file is the required file for training yolo-6D. It contains the hyperparameters, settings and so on.
 - weights are the pre-trained weights used in evaluation. 

Once run it, a .mat file will be generated in ".\backup\< obj-name >\" and the filename will be "prediction_< dataset >_< obj_name >.mat". (for example: "backup/hotstab/predictions_linemod_hotstab.mat") 

To compute metrics for yolo-6D predictions, you must create the csv files. 

```
python scripts/yolo6d2BOP.py -i <path-to-mat-file> -o <output-csv-filename> -t <original-test-txt-file-used-in-yolo6d-evaluation> -m <mode>
#python scripts/yolo6d2BOP.py -i ../../6d_pose_nn_repo/yolo-6d/results/res_tr_etutte_lr0005_bs32/predictions_linemod_hotstab.mat -o yolo6dpred_custom-test.csv -t ../../6d_pose_nn_repo/yolo-6d/yolo6D_ds/train/hotstab/test.txt -m pred
#python scripts/yolo6d2BOP.py -i ../../6d_pose_nn_repo/yolo-6d/results/res_tr_etutte_lr0005_bs32/predictions_linemod_hotstab.mat -o yolo6dgt_custom-test.csv -t ../../6d_pose_nn_repo/yolo-6d/yolo6D_ds/train/hotstab/test.txt -m gt
```
Where: 
 - path-to-mat-file: it is the mat file generated during the yolo-6D evaluation. It is generally in "< yolo6d-folder\backup\< obj-name >\prediction_< dataset >_< obj_name >.mat" (for example "/yolo-6d/backup/hotstab/predictions_linemod_hotstab.mat")
 - output-csv-filename : the desired output csv file name. it is recommended to use the following format: "< model >< mode >_custom-test.csv". (ex: yolo6dpred_custom-test.csv, yolo6dgt_custom-test.csv) 
 - original-test-txt-file-used-in-yolo6d-evaluation: path to the test.txt file in the yolo6D dataset. 
 - mode: type string. it specifies which file have to be generated: predictions or ground-truths. It can be "pred" or "gt". 

Then to compute the metrics, you can run: 
```
python scripts/est_pose_file2add_accuracy.py --result_filename <path-to-csv-estimation-file> --dataset_path <path-to-dataset-folder> --gt_filename <path-to-groundtruth-file> --gt_filetype <type>
#python scripts/est_pose_file2add_accuracy.py --result_filename ./yolo6dpred_custom-test.csv --dataset_path ./ --gt_filename ./yolo6dgt_custom-test.csv
#python scripts/est_pose_file2add_accuracy.py --result_filename /home/elena/repos/bop_toolkit/yolo6D_evaluation_files/yolo6dpred_custom-test.csv --dataset_path /home/elena/repos/bop_toolkit/yolo6D_evaluation_files --gt_filename /home/elena/repos/bop_toolkit/yolo6D_evaluation_files/yolo6dgt_custom-test.csv
```