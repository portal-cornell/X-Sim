# 🔄 Real-to-Sim Pipeline Documentation

## 📋 Overview

This pipeline converts real-world environments and objects into simulated representations for robotic manipulation and other applications.

---

## Part 1: Environment Scan 🌍

### 📸 2D Gaussian Splatting / Polycam

The environment scanning phase captures the spatial layout and visual appearance of the real-world scene.

**Tools:**
- 2D Gaussian Splatting for efficient scene representation
- [Polycam](https://poly.cam/) for photogrammetric reconstruction

**Process:**
1. Capture multiple images or video of the environment from various angles
2. Process the captured data to generate a 3D reconstruction
3. Export the environment model for simulation integration

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