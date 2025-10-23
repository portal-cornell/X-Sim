# 🔄 Real-to-Sim Pipeline Documentation

## 📋 Overview

This pipeline converts real-world environments and objects into simulated representations for robotic manipulation and other applications.

---

## Part 1: Environment Scan 🌍

### 📸 2D Gaussian Splatting / Polycam

The environment scanning phase captures the spatial layout and visual appearance of the real-world scene.

**Tools:**
- [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting) for efficient scene representation
- [Polycam](https://poly.cam/) for photogrammetric reconstruction
- Mesh editing software, e.g. [MeshLab](https://www.meshlab.net) and [Blender](https://www.blender.org)

**Generating the Mesh:**

Begin by cloning the above 2D Gaussian Splatting repository and cd into it. Create a conda environment using `conda env create --file environment.yml`. Then:
1. Capture many images of the environment from various angles
2. Put all photos of the scene into a ./datasets/<SCENE_NAME>/input
3. Activate your surfel_splatting conda environment
4. Process your images into a COLMAP dataset using `python convert.py -s datasets/<SCENE_NAME>`
5. Train your 2DG2 using `python train.py -s <path/to/COLMAP dataset>`. Take note of the name/ID of the model being trained
6. To render the mesh, use `python render.py -m <path/to/trained model> -s <path/to/COLMAP dataset>` where <path/to/trained model> is the directory in ./outputs with the name/ID from training
7. The output will be in ./output/<model_ID>/train/<step number, e.g. 30000>/fuse.ply

**Post Processing:**

We recommend first following the instructions at the following link to clean up your mesh in MeshLab: https://gist.github.com/shubhamwagh/0dc3b8173f662d39d4bf6f53d0f4d66b?permalink_comment_id=3721002

The scale and position/orientation of the mesh will be arbitrary and should be adjusted in MeshLab or Blender. You can open the application and then import the fuse.ply file (as .ply).

1. Begin by fixing the mesh scale. First, measure the full width of your scene in *meters*. Use the measure tool in Blender or MeshLab to measure the identical span. Since the orientation of the object is likely incorrect, it may be difficult to make sure the measure tool is not angled, so be careful.
- In MeshLab, you can use the measure tool with the textures active, so you can verify the line being drawn is parallel to any grain or surface texture, and the measurements are directly from the points clicked in the mesh.
- In Blender, you should snap the measure tool to the top of the table, otherwise you may end up measuring two points off of the surface.
In either tool, use the scale feature to scale the entire object with the ratio of (real-world measurement) / (mesh measurement).

2. For fixing the translation, we strongly advise to use Blender.
Click on the point of the kitchen you want to be the origin of your mesh to set your cursor. Be as precise as possible, and zoom in to make sure your selection is accurate.
Use Shift+S and select “Cursor to Selected.” This sets the origin of the object to your cursor location.

3. To set the rotation, we again strongly recommend using Blender.
In Blender, use the rotation tool. You will see several circles about the object’s origin (which should have been set with the above).
Drag the circles one at a time to set the orientation about each axis. For each axis of rotation, click on the outer edge of the circle & drag until the object is aligned to the real-world axes. You will likely have to repeat this process across the 3 axes several times as they each get more precise.
Check that the alignment is as close to perfect as can be by using 1, 3, 7, 9 on the NUMPAD. *Orthographic view* is necessary when verifying.

4. To export the mesh, use Blender. BEFORE EXPORTING: Make sure you “Apply Transform” (Ctrl+A or Object ‣ Apply ‣ Location / Rotation / Scale / Rotation & Scale) to the mesh. This sets the rotation/translation effects to the mesh locally instead of globally, ensuring that when we set the pose in simulation, the inputs don’t override the transforms set in Blender.
Make sure that the file you are viewing in Blender is textured: depending on how the object is loaded, this can be viewed by switching from object mode to Vertex Paint Mode.
Export using File > Export As > .glb or .gltf. Before finalizing the export, make sure Color is checked/selected on the right menu. This should output a single .gltf file with texture embedded.

---

## Part 2: Object Tracking 🎯

### 2.1 Object Scanning: Polycam 📦

Individual objects within the environment are scanned to create detailed 3D models.

**Tool:** [Polycam](https://poly.cam/)

**Process:**
1. Isolate the target object
2. Capture 360-degree coverage using Polycam
3. Generate a textured 3D mesh
4. Export the object model in a standard format (e.g., OBJ, PLY, GLTF)

---

### 2.2 Collecting Human Demos 🎬

Run `collect_human_demo.py` to collect demonstration data. We have provided code for the ZED camera.

**Inputs:**
- `serial_number`: ZED Camera serial number
- `save_dir`: Directory to save collected data

**Usage:**
```bash
python collect_human_demo.py --serial_number <SERIAL_NUMBER> --save_dir <SAVE_DIR>
```
---

### 2.3 Object Tracking: FoundationPose 🤖

[FoundationPose](https://github.com/NVlabs/FoundationPose) provides model-based 6DoF pose estimation for tracking objects in real-time.

#### 📥 Installation
```bash
git clone https://github.com/NVlabs/FoundationPose.git
cd FoundationPose
```

First follow the instructions to download the model weights in [Data Prepare](https://github.com/NVlabs/FoundationPose?tab=readme-ov-file#data-prepare). Feel free to download demo_data and play around with FoundationPose, we have provided our own sample data in the `sample_data` folder

We then followed [Env Setup Option 2: Conda (experimental)](https://github.com/NVlabs/FoundationPose?tab=readme-ov-file#env-setup-option-2-conda-experimental) to set up the environment.

**Common Setup Issues:**

During installation, you may encounter the following issues:
1. [Update to readme for cpp → python setup](https://github.com/NVlabs/FoundationPose/pull/185/files)
2. [Update C++ version to 17](https://github.com/NVlabs/FoundationPose/issues/35)
3. [Build without ninja](https://github.com/NVlabs/FoundationPose/issues/241)

#### 🚀 After Installation

Once installation and setup are complete, copy the two provided files into your FoundationPose directory:
```
FoundationPose/
├── fp_objects.py
├── generate_mask.py
```

**Command:**
```bash
mv fp_objects.py generate_mask.py FoundationPose/
```

#### 📋 Input Requirements

**`fp_objects.py`** requires three inputs:

| Parameter | Description |
|-----------|-------------|
| **`rgbd_frames_directory`** | Directory containing two subfolders:<br>• `rgb/` - RGB frames from your video<br>• `depth/` - Depth frames in uint16 format |
| **`object_meshes`** | Path(s) to mesh file(s) for tracking (`.obj` format) |
| **`camera_to_robot`** | `.npy` file containing a 4×4 transformation matrix from camera to robot frame |

**Usage:**
```bash
python fp_objects.py --rgbd_frames_directory <DEMO_DIR> --object_meshes <OBJ_1> <OBJ_2> --camera_to_robot <CAMERA_TO_ROBOT>
```

#### 🎥 Our Setup

- **Camera**: ZED 2i
