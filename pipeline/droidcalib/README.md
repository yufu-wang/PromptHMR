# DroidCalib

Code for the ICCV 2023 paper "Deep geometry-aware camera self-calibration from video" by Annika Hagemann, Moritz Knorr and Christoph Stiller. 

This codebase allows estimating camera intrinsics from monocular video without calibration targets. It is derived from DROID-SLAM by Teed et al. https://github.com/princeton-vl/DROID-SLAM and extends the deep visual SLAM system with a self-calibrating bundle adjustment layer.


![](/misc/DroidCalib.png)


So far, the repository contains a demo script for inference and code for reproducing the results from our paper. An extended version, containing the training code for self-calibration, will be made available soon.


## Purpose of the project
This software is a research prototype, solely developed for and published as part of the publication "Deep geometry-aware camera self-calibration from video", Hagemann et al. ICCV 2023. It will not be maintained or monitored.


## Requirements
To run the demo and for testing on your own sequence, you need a 12 GB GPU.
To reproduce the results from the paper, a 16 GB GPU is required. 


## Getting Started
1. Clone the additional thirdparty requirements:
    ```Bash
    git submodule update --init --recursive
    ```

2. Create an anaconda environment with all requirements:
    ```Bash
    conda env create -f environment_novis.yaml
    conda activate droidenv
    pip install evo --upgrade --no-binary evo
    ```
    If you get stuck at "Solving environment", try to use our detailed exported environment under misc/environment_detailed_vis.yaml (with visualization) or misc/environment_detailed.yaml (without visualization), instead of environment_novis.yaml.

3. Compile the extensions. This takes several minutes.
    ```Bash
    python setup.py install
    ```



## Running with an exemplary sequence
1. Download the exemplary sequence [abandonedfactory](https://drive.google.com/uc?id=1AlfhZnGmlsKWGcNHFB1i8i8Jzn4VHB15), unzip it and put it into the folder `datasets/demo`.

2. Run the demo script:
    ```Bash
    python demo.py --imagedir=datasets/demo/abandonedfactory --opt_intr --num_images=300
    ```
3. The estimated intrinsics will appear in the terminal. To output a video with estimated intrinsics, use the "--video_calib" flag when running the demo. Note that this slows down the inference. To run the 3D visualization, use the "--visualize" flag. Pressing the key "r" in the open3D viewer allows you to interact with the visualization during inference.

You can use the "--num_images" flag to adjust the number of images. For suitable sequences (diverse motion, structured environment), it is oftentimes sufficient to use around 300 images. Furthermore, you can adjust the image size to reduce computation time, and the stride to only use every n-th image.


## Reproducing main results from paper
1. Download the different datasets
* [TartanAir](https://tartanair.blob.core.windows.net/tartanair-testing1/tartanair-test-mono-release.tar.gz) monocular test sequences from the CVPR 2020 SLAM challenge with [groundtruth poses](https://cmu.box.com/shared/static/3p1sf0eljfwrz4qgbpc6g95xtn2alyfk.zip); put them into `datasets/TartanAir`
* [EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) sequences (ASL) and [groundtruth poses](https://github.com/princeton-vl/DROID-SLAM/tree/main/data/euroc_groundtruth); put them into `datasets/EuRoC`
* [TUM-RGBD](https://vision.in.tum.de/data/datasets/rgbd-dataset/download) fr1 sequences; put them into `datasets/TUM-RGBD`

    The expected folder structure of each dataset can be seen in the file evaluation_scripts/DroidCalib/config.yaml.

2. Run evaluation
    ```Bash
    python evaluation_scripts/DroidCalib/eval_script.py 
    ```
    This will create files figures/*.csv containing evaluation results with different settings.


## Running DroidCalib with your own data
You only need a monocular video, stored as an ordered set of images. If the images are distortion-free, use the pinhole model:
```Bash
python demo.py --imagedir=YOUR_IMAGE_PATH --opt_intr
```

To approximate radial distortion, we have implemented the unified camera model ("mei"):
```Bash
python demo.py --imagedir=YOUR_IMAGE_PATH --opt_intr --camera_model=mei
```
To estimate only the focal length, use the "focal" camera model:
```Bash
python demo.py --imagedir=YOUR_IMAGE_PATH --opt_intr --camera_model=focal
```
This can be useful in case the camera motion is not sufficiently diverse to render all intrinsic parameters observable (e.g. planar motion). Right now, this is only supported for distortion-free images.

### Hints to achieve an accurate calibration
* The video should contain diverse camera motion (translation and rotations around the different axes) for the intrinsics to be observable.
* Avoid motion blur and other artifacts in the images.
* There should be some structure in the scene (e.g. not just the sky or an unstructured wall).


## Acknowledgements
This repository is derived from DROID-SLAM by Teed et al. https://github.com/princeton-vl/DROID-SLAM -- we thank the authors for making their source code available. All files without header are left unchanged and originate from the authors of DROID-SLAM. Files with header were adapted for the self-calibration functionality.

## License
DroidCalib is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in DroidCalib, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).