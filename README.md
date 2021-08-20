# NeRF-Tex

This repo contains the implementation of NeRF-Tex: Neural Reflectance Field Textures written in [Tensorflow](https://www.tensorflow.org/). The paper and EGSR presentation can be found on the [project page](https://hbaatz.github.io/nerf-tex/). Should you have a question regarding the implementation feel free to contact [Hendrik Baatz](mailto:baatzh@student.ethz.ch).

# Setting up the code for use
You'll need to have [Conda](https://docs.conda.io/en/latest/) and [Embree 3](https://github.com/embree/embree) installed on your system. Then just follow the steps below with the working directory being the one where this README file resides.

1. Create & activate the python environment

    ```
    conda env create
    conda activate nerf-tex
    ```

2. Get the dependencies in case you did not already clone this repo with the --recursive flag

    ```
    git submodule update --init
    ```

3. Build the instancer code on Linux (you may want to adapt the Embree include and library dirs in [instancer/setup.py](instancer/setup.py) depending on your local setup)

    ```
    make -C instancer
    ```

# Creating the example dataset
We provide an example showing how our datasets are created. You'll need to have [Blender](https://www.blender.org/) (2.92) installed and in your path. This takes a while.
```
sh data/create_dataset.sh data/carpet.blend data/configs/config_carpet.py
python data/nerf2tfr.py datasets/materials/carpet dataset/materials/carpet/tfr
```

# Downloading the datasets
The datasets are available [here](https://drive.google.com/drive/folders/1xAvk1jewv7lGG25Iqd2-yh-CnicA30Iy?usp=sharing).

# Training an example scene
```
python main.py configs/config_carpet_train.py
```

# Rendering an example scene
```
python main.py configs/config_carpet_render.py
```

# Acknowledgements
We thank the contributors of the following projects used in this implentation for making their work available publicly:
- [The original NeRF implementation by Ben Mildenhall](https://github.com/bmild/nerf)
- [Embree](https://github.com/embree/embree)
- [libigl](https://github.com/libigl/libigl.git)
- [eigen](https://eigen.tuxfamily.org)
- [stb](https://github.com/nothings/stb.git)
- [nlohmann/json](https://github.com/nlohmann/json.git)
- [The Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/)