import os
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import argparse
from tqdm import tqdm
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.animation as animation
from matplotlib.patches import Circle
import cv2
from IPython.display import HTML
# from train import load_model
from .dp_training_rgb import load_model
import re
import json
import torch.nn.functional as F

def setup_encoder(checkpoint_path: str, device: str):
    """
    Load a pre-trained policy model and get its encoder.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        The policy model (which contains the encoder)
    """
    # Load the policy model using the imported function
    policy, dataset, train_cfg = load_model(checkpoint_path, device)
    
    # Set the model to evaluation mode
    policy.eval()
    
    return policy, dataset

def process_npz_file(
    npz_path: str, 
    policy, 
    dataset,
    device: str, 
    image_key: str = 'zed_sim_images'
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Load images from an NPZ file, encode them, and store the embeddings.
    Processes the entire episode at once for maximum efficiency.
    
    Args:
        npz_path: Path to the NPZ file
        policy: The policy model containing the encoder
        dataset: The dataset object used for processing observations
        device: Device to use for processing ('cuda' or 'cpu')
        image_key: Key for the images in the NPZ file
        
    Returns:
        Tuple of (Dictionary with original data and added embeddings, raw embeddings array, original images array)
    """
    print(f"Processing NPZ file: {npz_path}")
    
    # Load the NPZ file
    data = np.load(npz_path, allow_pickle=True)["episode"]
    
    # List to store all processed observations and original images
    all_processed_obs = []
    original_images = []

    # First, preprocess all observations and store them
    print(f"Preprocessing {len(data)} timesteps...")
    for timestep in data:
        obs_dict = timestep['obs']
        # Extract and store original images before processing
        if image_key in obs_dict:
            original_images.append(obs_dict[image_key])
            
        # Process observation but don't move to GPU or add batch dimension yet
        processed_obs = dataset.process_observation(obs_dict)
        all_processed_obs.append(processed_obs)

    # Convert original images to numpy array
    if original_images:
        original_images = np.array(original_images)
    else:
        print(f"Warning: No images found with key '{image_key}'. Using placeholder images.")
        # Create placeholder images if none found
        original_images = np.zeros((len(data), 256, 256, 3), dtype=np.uint8)

    # Process the entire episode at once
    with torch.no_grad():
        # Initialize batch dictionary with empty tensors
        batch_dict = {}
        
        # Get keys from the first item
        for key in all_processed_obs[0].keys():
            # Stack all tensors with the same key from all items in the batch
            stacked_tensors = torch.stack([obs[key] for obs in all_processed_obs])
            batch_dict[key] = stacked_tensors.to(device)
        # Get embeddings for the entire episode at once
        embeddings = policy.net.encoder(batch_dict).cpu().numpy()
    
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Create a copy of the data with embeddings added
    result_data = data.copy()
    
    # Add embeddings to each timestep
    for i, timestep in enumerate(result_data):
        timestep['embedding'] = embeddings[i]
    
    return result_data, embeddings, original_images

def compute_tsne(embeddings, n_components=2, perplexity=5, n_iter=1000, random_state=42):
    """
    Compute t-SNE dimensionality reduction on embeddings.
    
    Args:
        embeddings: Numpy array of embeddings (N x embedding_dim)
        n_components: Number of dimensions to reduce to (typically 2 or 3)
        perplexity: Perplexity parameter for t-SNE
        n_iter: Number of iterations for t-SNE
        random_state: Random seed for reproducibility
        
    Returns:
        Reduced embeddings (N x n_components)
    """
    print(f"Computing t-SNE with {n_components} components...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                n_iter=n_iter, random_state=random_state)
    return tsne.fit_transform(embeddings)

def create_live_tsne_animation(
    tsne_embeddings: np.ndarray,
    original_images: np.ndarray,
    output_path: str,
    title: str = "t-SNE Visualization with Live Video Frames",
    fps: int = 10,
    dpi: int = 150,
    marker_size: int = 8,
    highlight_current: bool = True
):
    """
    Create an animation that shows both the t-SNE plot and the corresponding video frame.
    
    Args:
        tsne_embeddings: Numpy array of t-SNE embeddings (N x 2) or (N x 3)
        original_images: Numpy array of original video frames (N x H x W x C)
        output_path: Path to save the animation (MP4 file)
        title: Title for the animation
        fps: Frames per second for the animation
        dpi: DPI for the animation
        marker_size: Size of markers in the t-SNE plot
        highlight_current: Whether to highlight the current point
        trail_length: Number of previous points to show in the trail
    """
    print(f"Creating live t-SNE animation with {len(tsne_embeddings)} frames...")
    
    # Set up the figure with two subplots: one for t-SNE, one for the image
    fig = plt.figure(figsize=(16, 8))
    
    # Create subplots
    if tsne_embeddings.shape[1] == 3:
        ax1 = fig.add_subplot(121, projection='3d')
    else:
        ax1 = fig.add_subplot(121)
    
    ax2 = fig.add_subplot(122)
    
    # Create a colormap for the points
    cmap = plt.cm.viridis
    norm = Normalize(0, 1)
    
    # Pre-process the original images if needed
    # If images are in range [0, 1], scale to [0, 255]
    if original_images.dtype == np.float32 or original_images.dtype == np.float64:
        if np.max(original_images) <= 1.0:
            original_images = (original_images * 255).astype(np.uint8)
    
    # Initialize animation
    point, = ax1.plot([], [], 'o', markersize=marker_size*2, color='red')
    img = ax2.imshow(np.zeros_like(original_images[0]), animated=True)
    
    # Add title
    fig.suptitle(title, fontsize=16)
    ax1.set_title("t-SNE Embedding Space")
    ax2.set_title("Current Video Frame")
    
    # Set up axes for t-SNE plot
    if tsne_embeddings.shape[1] == 3:
        ax1.set_xlabel("t-SNE Component 1")
        ax1.set_ylabel("t-SNE Component 2")
        ax1.set_zlabel("t-SNE Component 3")
    else:
        ax1.set_xlabel("t-SNE Component 1")
        ax1.set_ylabel("t-SNE Component 2")
    
    # Remove ticks from image plot
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Initialize point to highlight current position
    highlight_point = None
    
    # Set up axes limits to show the entire t-SNE plot
    if tsne_embeddings.shape[1] == 3:
        x_min, x_max = tsne_embeddings[:, 0].min(), tsne_embeddings[:, 0].max()
        y_min, y_max = tsne_embeddings[:, 1].min(), tsne_embeddings[:, 1].max()
        z_min, z_max = tsne_embeddings[:, 2].min(), tsne_embeddings[:, 2].max()
        
        # Add padding
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        z_padding = (z_max - z_min) * 0.1
        
        ax1.set_xlim(x_min - x_padding, x_max + x_padding)
        ax1.set_ylim(y_min - y_padding, y_max + y_padding)
        ax1.set_zlim(z_min - z_padding, z_max + z_padding)
    else:
        x_min, x_max = tsne_embeddings[:, 0].min(), tsne_embeddings[:, 0].max()
        y_min, y_max = tsne_embeddings[:, 1].min(), tsne_embeddings[:, 1].max()
        
        # Add padding
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        
        ax1.set_xlim(x_min - x_padding, x_max + x_padding)
        ax1.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Plot all t-SNE points with low opacity to show the entire space
    if tsne_embeddings.shape[1] == 3:
        ax1.scatter(
            tsne_embeddings[:, 0],
            tsne_embeddings[:, 1],
            tsne_embeddings[:, 2],
            color='gray',
            alpha=0.2,
            s=marker_size
        )
    else:
        ax1.scatter(
            tsne_embeddings[:, 0],
            tsne_embeddings[:, 1],
            color='gray',
            alpha=0.2,
            s=marker_size
        )
    
    # Function to initialize the animation
    def init():
        if tsne_embeddings.shape[1] == 3:
            point.set_data([], [])
            point.set_3d_properties([])
        else:
            point.set_data([], [])
        
        img.set_array(np.zeros_like(original_images[0]))
        return point, img
    
    # Function to update the animation for each frame
    def update(frame):
        # Update the current point in the t-SNE plot
        current_point = tsne_embeddings[frame]
        
        # Update the current point marker
        if tsne_embeddings.shape[1] == 3:
            point.set_data([current_point[0]], [current_point[1]])
            point.set_3d_properties([current_point[2]])
        else:
            point.set_data([current_point[0]], [current_point[1]])
        
        # Update the image
        img.set_array(original_images[frame])
        
        # Add frame counter
        ax2.set_xlabel(f"Frame: {frame+1}/{len(tsne_embeddings)}")
        
        # Draw trajectory line up to current frame
        trajectory = None
        if frame > 0:
            if tsne_embeddings.shape[1] == 3:
                trajectory = ax1.plot(
                    tsne_embeddings[:frame+1, 0],
                    tsne_embeddings[:frame+1, 1],
                    tsne_embeddings[:frame+1, 2],
                    'b-',
                    alpha=0.5,
                    linewidth=1
                )[0]
            else:
                trajectory = ax1.plot(
                    tsne_embeddings[:frame+1, 0],
                    tsne_embeddings[:frame+1, 1],
                    'b-',
                    alpha=0.5,
                    linewidth=1
                )[0]
            
        # Add a highlight circle around the current point if requested
        highlight = None
        if highlight_current and tsne_embeddings.shape[1] == 2:
            highlight = Circle(
                (current_point[0], current_point[1]),
                radius=marker_size/30,
                fill=False,
                edgecolor='red',
                linewidth=2
            )
            ax1.add_patch(highlight)
        
        return_elements = [point, img]
        if trajectory:
            return_elements.append(trajectory)
        if highlight:
            return_elements.append(highlight)
        
        return tuple(return_elements)
    
    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(tsne_embeddings),
        init_func=init, blit=False, interval=1000/fps
    )
    
    # Set up the writer
    writer = animation.FFMpegWriter(fps=fps)
    
    # Save the animation
    print(f"Saving animation to: {output_path}")
    ani.save(output_path, writer=writer, dpi=dpi)
    
    plt.close(fig)
    print(f"Animation saved successfully!")
    
    return output_path

def create_interactive_tsne_display(
    tsne_embeddings: np.ndarray,
    original_images: np.ndarray,
    output_dir: str,
    base_filename: str,
    title: str = "Interactive t-SNE Visualization",
    sample_rate: int = 1,  # Sample rate for saving individual frames
    create_html: bool = True  # Whether to create an HTML file for interactive viewing
):
    """
    Create an interactive display with t-SNE points and corresponding video frames.
    Saves individual frames that can be browsed interactively.
    
    Args:
        tsne_embeddings: Numpy array of t-SNE embeddings (N x 2) or (N x 3)
        original_images: Numpy array of original video frames (N x H x W x C)
        output_dir: Directory to save the output files
        base_filename: Base filename for output files
        title: Title for the visualization
        sample_rate: Sample rate for saving individual frames (1 means save every frame)
        create_html: Whether to create an HTML file for interactive viewing
    """
    print(f"Creating interactive t-SNE display...")
    
    # Create output directory for frames
    frames_dir = os.path.join(output_dir, f"{base_filename}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Calculate point colors based on time progression
    n_points = len(tsne_embeddings)
    colors = plt.cm.viridis(np.linspace(0, 1, n_points))
    
    # Calculate plot limits with padding
    if tsne_embeddings.shape[1] == 3:
        x_min, x_max = tsne_embeddings[:, 0].min(), tsne_embeddings[:, 0].max()
        y_min, y_max = tsne_embeddings[:, 1].min(), tsne_embeddings[:, 1].max()
        z_min, z_max = tsne_embeddings[:, 2].min(), tsne_embeddings[:, 2].max()
        
        # Add padding
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        z_padding = (z_max - z_min) * 0.1
        
        x_limits = (x_min - x_padding, x_max + x_padding)
        y_limits = (y_min - y_padding, y_max + y_padding)
        z_limits = (z_min - z_padding, z_max + z_padding)
    else:
        x_min, x_max = tsne_embeddings[:, 0].min(), tsne_embeddings[:, 0].max()
        y_min, y_max = tsne_embeddings[:, 1].min(), tsne_embeddings[:, 1].max()
        
        # Add padding
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        
        x_limits = (x_min - x_padding, x_max + x_padding)
        y_limits = (y_min - y_padding, y_max + y_padding)
    
    # Create the static t-SNE plot with all points
    plt.figure(figsize=(10, 8))
    
    if tsne_embeddings.shape[1] == 3:
        ax = plt.subplot(111, projection='3d')
        scatter = ax.scatter(
            tsne_embeddings[:, 0],
            tsne_embeddings[:, 1],
            tsne_embeddings[:, 2],
            c=np.arange(n_points),
            cmap='viridis',
            s=30,
            alpha=0.7
        )
        
        # Set consistent limits
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_zlim(z_limits)
    else:
        ax = plt.subplot(111)
        scatter = ax.scatter(
            tsne_embeddings[:, 0],
            tsne_embeddings[:, 1],
            c=np.arange(n_points),
            cmap='viridis',
            s=30,
            alpha=0.7
        )
        
        # Set consistent limits
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Time progression')
    
    # Highlight start and end points
    ax.scatter(
        tsne_embeddings[0, 0],
        tsne_embeddings[0, 1],
        *([] if tsne_embeddings.shape[1] == 2 else [tsne_embeddings[0, 2]]),
        color='green',
        s=100,
        label='Start',
        edgecolors='black'
    )
    ax.scatter(
        tsne_embeddings[-1, 0],
        tsne_embeddings[-1, 1],
        *([] if tsne_embeddings.shape[1] == 2 else [tsne_embeddings[-1, 2]]),
        color='red',
        s=100,
        label='End',
        edgecolors='black'
    )
    
    # Draw the full trajectory
    if tsne_embeddings.shape[1] == 3:
        ax.plot(
            tsne_embeddings[:, 0],
            tsne_embeddings[:, 1],
            tsne_embeddings[:, 2],
            'gray',
            alpha=0.5,
            linewidth=1
        )
    else:
        ax.plot(
            tsne_embeddings[:, 0],
            tsne_embeddings[:, 1],
            'gray',
            alpha=0.5,
            linewidth=1
        )
    
    plt.title(f"{title} - All Points")
    plt.legend()
    
    # Save the static plot
    static_plot_path = os.path.join(output_dir, f"{base_filename}_static.png")
    plt.savefig(static_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save individual frames with highlighted current point
    html_content = []
    if create_html:
        html_content.append(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ display: flex; flex-wrap: wrap; }}
                .tsne-container {{ flex: 1; min-width: 500px; }}
                .frame-container {{ flex: 1; min-width: 500px; }}
                .controls {{ margin: 20px 0; }}
                img {{ max-width: 100%; }}
                #frameSlider {{ width: 80%; }}
                #frameDisplay {{ font-weight: bold; margin-left: 10px; }}
                .play-button {{ padding: 8px 16px; margin-right: 10px; cursor: pointer; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <div class="controls">
                <button id="playButton" class="play-button">Play</button>
                <input type="range" id="frameSlider" min="0" max="{n_points-1}" value="0">
                <span id="frameDisplay">Frame: 1/{n_points}</span>
            </div>
            <div class="container">
                <div class="tsne-container">
                    <h2>t-SNE Embedding Space</h2>
                    <img id="tsneImg" src="{base_filename}_frames/frame_0.png">
                </div>
                <div class="frame-container">
                    <h2>Video Frame</h2>
                    <img id="videoImg" src="{base_filename}_frames/video_0.png">
                </div>
            </div>
            <script>
                const slider = document.getElementById('frameSlider');
                const frameDisplay = document.getElementById('frameDisplay');
                const tsneImg = document.getElementById('tsneImg');
                const videoImg = document.getElementById('videoImg');
                const playButton = document.getElementById('playButton');
                const maxFrame = {n_points-1};
                let isPlaying = false;
                let playInterval;
                
                // Update displays based on current frame
                function updateFrame(frame) {{
                    frameDisplay.textContent = `Frame: ${{frame+1}}/${{maxFrame+1}}`;
                    tsneImg.src = `{base_filename}_frames/frame_${{frame}}.png`;
                    videoImg.src = `{base_filename}_frames/video_${{frame}}.png`;
                    slider.value = frame;
                }}
                
                // Handle slider change
                slider.addEventListener('input', function() {{
                    const frame = parseInt(this.value);
                    updateFrame(frame);
                    if (isPlaying) {{
                        stopPlayback();
                    }}
                }});
                
                // Play button functionality
                playButton.addEventListener('click', function() {{
                    if (isPlaying) {{
                        stopPlayback();
                    }} else {{
                        startPlayback();
                    }}
                }});
                
                function startPlayback() {{
                    isPlaying = true;
                    playButton.textContent = 'Pause';
                    let currentFrame = parseInt(slider.value);
                    
                    playInterval = setInterval(() => {{
                        currentFrame++;
                        if (currentFrame > maxFrame) {{
                            currentFrame = 0;
                        }}
                        updateFrame(currentFrame);
                    }}, 200); // Adjust speed as needed
                }}
                
                function stopPlayback() {{
                    isPlaying = false;
                    playButton.textContent = 'Play';
                    clearInterval(playInterval);
                }}
            </script>
        </body>
        </html>
        """)
    
    # Save frames at the specified sample rate
    for i in tqdm(range(0, n_points, sample_rate), desc="Saving frames"):
        # Create t-SNE plot with current point highlighted
        plt.figure(figsize=(10, 8))
        
        if tsne_embeddings.shape[1] == 3:
            ax = plt.subplot(111, projection='3d')
            # Plot all points with lower alpha
            ax.scatter(
                tsne_embeddings[:, 0],
                tsne_embeddings[:, 1],
                tsne_embeddings[:, 2],
                c='lightgray',
                s=20,
                alpha=0.3
            )
            
            # Set consistent limits
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
            ax.set_zlim(z_limits)
            
            # Draw trajectory up to current point
            if i > 0:
                ax.plot(
                    tsne_embeddings[:i+1, 0],
                    tsne_embeddings[:i+1, 1],
                    tsne_embeddings[:i+1, 2],
                    'b-',
                    alpha=0.6,
                    linewidth=1.5
                )
            
            # Highlight current point
            ax.scatter(
                tsne_embeddings[i, 0],
                tsne_embeddings[i, 1],
                tsne_embeddings[i, 2],
                color='red',
                s=100,
                edgecolors='black'
            )
        else:
            ax = plt.subplot(111)
            # Plot all points with lower alpha
            ax.scatter(
                tsne_embeddings[:, 0],
                tsne_embeddings[:, 1],
                c='lightgray',
                s=20,
                alpha=0.3
            )
            
            # Set consistent limits
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
            
            # Draw trajectory up to current point
            if i > 0:
                ax.plot(
                    tsne_embeddings[:i+1, 0],
                    tsne_embeddings[:i+1, 1],
                    'b-',
                    alpha=0.6,
                    linewidth=1.5
                )
            
            # Highlight current point
            ax.scatter(
                tsne_embeddings[i, 0],
                tsne_embeddings[i, 1],
                color='red',
                s=100,
                edgecolors='black'
            )
        
        # Label start and end points
        if i == 0:
            ax.text(
                tsne_embeddings[0, 0],
                tsne_embeddings[0, 1],
                *([] if tsne_embeddings.shape[1] == 2 else [tsne_embeddings[0, 2]]),
                "Start",
                fontsize=10,
                verticalalignment='bottom'
            )
        
        plt.title(f"t-SNE - Frame {i+1}/{n_points}")
        
        # Save t-SNE frame
        tsne_frame_path = os.path.join(frames_dir, f"frame_{i}.png")
        plt.savefig(tsne_frame_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save corresponding video frame
        video_frame = original_images[i]
        # Ensure the image is in the right format for saving
        if video_frame.dtype == np.float32 or video_frame.dtype == np.float64:
            if np.max(video_frame) <= 1.0:
                video_frame = (video_frame * 255).astype(np.uint8)
        
        video_frame_path = os.path.join(frames_dir, f"video_{i}.png")
        # Use OpenCV to save the image
        cv2.imwrite(video_frame_path, cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR))
    
    # Save HTML file if requested
    if create_html and html_content:
        html_path = os.path.join(output_dir, f"{base_filename}_interactive.html")
        with open(html_path, 'w') as f:
            f.write(html_content[0])
        print(f"Interactive HTML saved to: {html_path}")
    
    return {
        'static_plot': static_plot_path,
        'frames_dir': frames_dir,
        'html_path': os.path.join(output_dir, f"{base_filename}_interactive.html") if create_html else None
    }

def save_embeddings(output_path: str, data: Dict[str, np.ndarray]):
    """
    Save the data dictionary with embeddings to a new NPZ file.
    
    Args:
        output_path: Path to save the output NPZ file
        data: Dictionary with data and embeddings
    """
    print(f"Saving embeddings to: {output_path}")
    np.savez_compressed(output_path, **data)

def process_directory(
    input_dir: str,
    output_dir: str,
    policy,
    dataset,
    device: str, 
    image_key: str = 'zed_sim_images',
    visualize_tsne: bool = True,
    tsne_components: int = 2,
    tsne_perplexity: int = 5,
    tsne_iterations: int = 1000,
    create_animation: bool = True,
    create_interactive: bool = True,
    animation_fps: int = 10,
    sample_rate: int = 1
):
    """
    Process all NPZ files in a directory.
    
    Args:
        input_dir: Directory containing NPZ files
        output_dir: Directory to save processed files
        policy: The policy model containing the encoder
        dataset: The dataset object for processing observations
        device: Device to use for processing
        image_key: Key for the images in the NPZ files
        visualize_tsne: Whether to generate t-SNE visualizations
        tsne_components: Number of components for t-SNE (2 or 3)
        tsne_perplexity: Perplexity parameter for t-SNE
        tsne_iterations: Number of iterations for t-SNE
        create_animation: Whether to create animated visualizations
        create_interactive: Whether to create interactive HTML visualizations
        animation_fps: Frames per second for animations
        sample_rate: Sample rate for interactive visualization frames
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create directories for visualizations if needed
    if visualize_tsne:
        vis_output_path = output_path / "visualizations"
        vis_output_path.mkdir(parents=True, exist_ok=True)
        
        if create_animation:
            anim_output_path = vis_output_path / "animations"
            anim_output_path.mkdir(parents=True, exist_ok=True)
        
        if create_interactive:
            interactive_output_path = vis_output_path / "interactive"
            interactive_output_path.mkdir(parents=True, exist_ok=True)
    
    npz_files = list(input_path.glob('*.npz'))
    print(f"Found {len(npz_files)} NPZ files to process")
    
    for npz_file in tqdm(npz_files, desc="Processing NPZ files"):
        try:
            output_file = output_path / f"{npz_file.stem}_with_embeddings.npz"
            data, embeddings, original_images = process_npz_file(str(npz_file), policy, dataset, device, image_key)
            save_embeddings(str(output_file), {"episode": data})
            print(f"Successfully processed: {npz_file.name}")
            
            # Generate t-SNE visualization if requested
            if visualize_tsne:
                # Compute t-SNE
                tsne_embeddings = compute_tsne(
                    embeddings, 
                    n_components=tsne_components,
                    perplexity=tsne_perplexity,
                    n_iter=tsne_iterations
                )
                
                # Create animation if requested
                if create_animation:
                    animation_file = anim_output_path / f"{npz_file.stem}_tsne_animation.mp4"
                    create_live_tsne_animation(
                        tsne_embeddings,
                        original_images,
                        str(animation_file),
                        title=f"t-SNE Visualization - {npz_file.stem}",
                        fps=animation_fps
                    )
                
                # Create interactive visualization if requested
                if create_interactive:
                    create_interactive_tsne_display(
                        tsne_embeddings,
                        original_images,
                        str(interactive_output_path),
                        npz_file.stem,
                        title=f"t-SNE Visualization - {npz_file.stem}",
                        sample_rate=sample_rate
                    )
        except Exception as e:
            print(f"Error processing {npz_file.name}: {e}")

def process_npz_files_for_comparison(
    npz_path1: str,
    npz_path2: str, 
    policy,
    dataset,
    device: str, 
    image_key: str = 'zed_sim_images'
) -> Tuple[Dict, np.ndarray, np.ndarray, Dict, np.ndarray, np.ndarray]:
    """
    Load images from two NPZ files, encode them, and store the embeddings.
    
    Args:
        npz_path1: Path to the first NPZ file
        npz_path2: Path to the second NPZ file
        policy: The policy model containing the encoder
        dataset: The dataset object used for processing observations
        device: Device to use for processing ('cuda' or 'cpu')
        image_key: Key for the images in the NPZ file
        
    Returns:
        Tuple of (Dictionaries and arrays for both files)
    """
    print(f"Processing NPZ files for comparison:")
    print(f"File 1: {npz_path1}")
    print(f"File 2: {npz_path2}")
    
    # Process first file
    data1, embeddings1, original_images1 = process_npz_file(
        npz_path1, policy, dataset, device, image_key
    )
    
    # Process second file
    data2, embeddings2, original_images2 = process_npz_file(
        npz_path2, policy, dataset, device, image_key
    )
    
    # Check that both files have the same length
    if len(embeddings1) != len(embeddings2):
        print(f"Warning: Files have different lengths. File 1: {len(embeddings1)}, File 2: {len(embeddings2)}")
        print("Will proceed with comparison but visualizations may be affected.")
    
    return data1, embeddings1, original_images1, data2, embeddings2, original_images2

def compute_combined_tsne(
    embeddings1: np.ndarray, 
    embeddings2: np.ndarray, 
    n_components=2, 
    perplexity=5, 
    n_iter=1000, 
    random_state=42
):
    """
    Compute t-SNE dimensionality reduction on combined embeddings from two sources.
    
    Args:
        embeddings1: Numpy array of embeddings from first file (N x embedding_dim)
        embeddings2: Numpy array of embeddings from second file (M x embedding_dim)
        n_components: Number of dimensions to reduce to (typically 2 or 3)
        perplexity: Perplexity parameter for t-SNE
        n_iter: Number of iterations for t-SNE
        random_state: Random seed for reproducibility
        
    Returns:
        Combined reduced embeddings and indices to separate the sources
    """
    print(f"Computing combined t-SNE with {n_components} components...")
    
    # Combine embeddings
    combined_embeddings = np.vstack([embeddings1, embeddings2])
    
    # Compute t-SNE on combined embeddings
    tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                n_iter=n_iter, random_state=random_state)
    combined_tsne = tsne.fit_transform(combined_embeddings)
    
    # Create indices to separate the two sources
    indices1 = np.arange(len(embeddings1))
    indices2 = np.arange(len(embeddings1), len(embeddings1) + len(embeddings2))
    
    # Extract t-SNE results for each source
    tsne1 = combined_tsne[indices1]
    tsne2 = combined_tsne[indices2]
    
    return tsne1, tsne2, combined_tsne

def create_comparison_tsne_animation(
    tsne_embeddings1: np.ndarray,
    tsne_embeddings2: np.ndarray,
    original_images1: np.ndarray,
    original_images2: np.ndarray,
    output_path: str,
    title: str = "Comparison t-SNE Visualization",
    fps: int = 10,
    dpi: int = 150,
    marker_size: int = 8,
    highlight_current: bool = True
):
    """
    Create an animation that shows both the comparison t-SNE plot and the corresponding video frames.
    
    Args:
        tsne_embeddings1: Numpy array of t-SNE embeddings for first file (N x 2) or (N x 3)
        tsne_embeddings2: Numpy array of t-SNE embeddings for second file (M x 2) or (M x 3)
        original_images1: Numpy array of original video frames for first file (N x H x W x C)
        original_images2: Numpy array of original video frames for second file (M x H x W x C)
        output_path: Path to save the animation (MP4 file)
        title: Title for the animation
        fps: Frames per second for the animation
        dpi: DPI for the animation
        marker_size: Size of markers in the t-SNE plot
        highlight_current: Whether to highlight the current point
    """
    print(f"Creating comparison t-SNE animation...")
    
    # Get the minimum length to ensure we don't go out of bounds
    min_length = min(len(tsne_embeddings1), len(tsne_embeddings2))
    
    # Set up the figure with three subplots: one for t-SNE, two for the images
    fig = plt.figure(figsize=(18, 8))
    
    # Create grid spec for better layout control
    gs = fig.add_gridspec(1, 3, width_ratios=[2, 1, 1])
    
    # Create subplots
    if tsne_embeddings1.shape[1] == 3:
        ax1 = fig.add_subplot(gs[0], projection='3d')
    else:
        ax1 = fig.add_subplot(gs[0])
    
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    # Create colormaps for the trajectories
    cmap1 = plt.cm.Blues
    cmap2 = plt.cm.Oranges
    
    # Pre-process the original images if needed
    for images in [original_images1, original_images2]:
        if images.dtype == np.float32 or images.dtype == np.float64:
            if np.max(images) <= 1.0:
                images = (images * 255).astype(np.uint8)
    
    # Initialize animation elements
    point1, = ax1.plot([], [], 'o', markersize=marker_size*2, color='blue')
    point2, = ax1.plot([], [], 'o', markersize=marker_size*2, color='orange')
    img1 = ax2.imshow(np.zeros_like(original_images1[0]), animated=True)
    img2 = ax3.imshow(np.zeros_like(original_images2[0]), animated=True)
    
    # Add titles
    fig.suptitle(title, fontsize=16)
    ax1.set_title("t-SNE Embedding Space Comparison")
    ax2.set_title("File 1 Video Frame")
    ax3.set_title("File 2 Video Frame")
    
    # Set up axes for t-SNE plot
    if tsne_embeddings1.shape[1] == 3:
        ax1.set_xlabel("t-SNE Component 1")
        ax1.set_ylabel("t-SNE Component 2")
        ax1.set_zlabel("t-SNE Component 3")
    else:
        ax1.set_xlabel("t-SNE Component 1")
        ax1.set_ylabel("t-SNE Component 2")
    
    # Remove ticks from image plots
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    # Set up axes limits to show the entire t-SNE plot (combining both datasets)
    combined_embeddings = np.vstack([tsne_embeddings1, tsne_embeddings2])
    
    if tsne_embeddings1.shape[1] == 3:
        x_min, x_max = combined_embeddings[:, 0].min(), combined_embeddings[:, 0].max()
        y_min, y_max = combined_embeddings[:, 1].min(), combined_embeddings[:, 1].max()
        z_min, z_max = combined_embeddings[:, 2].min(), combined_embeddings[:, 2].max()
        
        # Add padding
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        z_padding = (z_max - z_min) * 0.1
        
        ax1.set_xlim(x_min - x_padding, x_max + x_padding)
        ax1.set_ylim(y_min - y_padding, y_max + y_padding)
        ax1.set_zlim(z_min - z_padding, z_max + z_padding)
    else:
        x_min, x_max = combined_embeddings[:, 0].min(), combined_embeddings[:, 0].max()
        y_min, y_max = combined_embeddings[:, 1].min(), combined_embeddings[:, 1].max()
        
        # Add padding
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        
        ax1.set_xlim(x_min - x_padding, x_max + x_padding)
        ax1.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Plot all t-SNE points with low opacity to show the entire space
    if tsne_embeddings1.shape[1] == 3:
        # Plot file 1 points
        ax1.scatter(
            tsne_embeddings1[:, 0],
            tsne_embeddings1[:, 1],
            tsne_embeddings1[:, 2],
            color='blue',
            alpha=0.2,
            s=marker_size,
            label='File 1'
        )
        # Plot file 2 points
        ax1.scatter(
            tsne_embeddings2[:, 0],
            tsne_embeddings2[:, 1],
            tsne_embeddings2[:, 2],
            color='orange',
            alpha=0.2,
            s=marker_size,
            label='File 2'
        )
    else:
        # Plot file 1 points
        ax1.scatter(
            tsne_embeddings1[:, 0],
            tsne_embeddings1[:, 1],
            color='blue',
            alpha=0.2,
            s=marker_size,
            label='File 1'
        )
        # Plot file 2 points
        ax1.scatter(
            tsne_embeddings2[:, 0],
            tsne_embeddings2[:, 1],
            color='orange',
            alpha=0.2,
            s=marker_size,
            label='File 2'
        )
    
    # Add legend
    ax1.legend()
    
    # Function to initialize the animation
    def init():
        if tsne_embeddings1.shape[1] == 3:
            point1.set_data([], [])
            point1.set_3d_properties([])
            point2.set_data([], [])
            point2.set_3d_properties([])
        else:
            point1.set_data([], [])
            point2.set_data([], [])
        
        img1.set_array(np.zeros_like(original_images1[0]))
        img2.set_array(np.zeros_like(original_images2[0]))
        return point1, point2, img1, img2
    
    # Function to update the animation for each frame
    def update(frame):
        # Ensure we don't go out of bounds
        frame = min(frame, min_length - 1)
        
        # Update the current points in the t-SNE plot
        current_point1 = tsne_embeddings1[frame]
        current_point2 = tsne_embeddings2[frame]
        
        # Update the current point markers
        if tsne_embeddings1.shape[1] == 3:
            point1.set_data([current_point1[0]], [current_point1[1]])
            point1.set_3d_properties([current_point1[2]])
            point2.set_data([current_point2[0]], [current_point2[1]])
            point2.set_3d_properties([current_point2[2]])
        else:
            point1.set_data([current_point1[0]], [current_point1[1]])
            point2.set_data([current_point2[0]], [current_point2[1]])
        
        # Update the images
        img1.set_array(original_images1[frame])
        img2.set_array(original_images2[frame])
        
        # Add frame counter
        ax1.set_xlabel(f"t-SNE Component 1 (Frame: {frame+1}/{min_length})")
        
        # Draw trajectory lines up to current frame
        trajectory1 = None
        trajectory2 = None
        
        if frame > 0:
            if tsne_embeddings1.shape[1] == 3:
                trajectory1 = ax1.plot(
                    tsne_embeddings1[:frame+1, 0],
                    tsne_embeddings1[:frame+1, 1],
                    tsne_embeddings1[:frame+1, 2],
                    'b-',
                    alpha=0.5,
                    linewidth=1
                )[0]
                
                trajectory2 = ax1.plot(
                    tsne_embeddings2[:frame+1, 0],
                    tsne_embeddings2[:frame+1, 1],
                    tsne_embeddings2[:frame+1, 2],
                    color='orange',
                    alpha=0.5,
                    linewidth=1
                )[0]
            else:
                trajectory1 = ax1.plot(
                    tsne_embeddings1[:frame+1, 0],
                    tsne_embeddings1[:frame+1, 1],
                    'b-',
                    alpha=0.5,
                    linewidth=1
                )[0]
                
                trajectory2 = ax1.plot(
                    tsne_embeddings2[:frame+1, 0],
                    tsne_embeddings2[:frame+1, 1],
                    color='orange',
                    alpha=0.5,
                    linewidth=1
                )[0]
            
        # Add highlight circles around the current points if requested
        highlight1 = None
        highlight2 = None
        
        if highlight_current and tsne_embeddings1.shape[1] == 2:
            highlight1 = Circle(
                (current_point1[0], current_point1[1]),
                radius=marker_size/30,
                fill=False,
                edgecolor='blue',
                linewidth=2
            )
            ax1.add_patch(highlight1)
            
            highlight2 = Circle(
                (current_point2[0], current_point2[1]),
                radius=marker_size/30,
                fill=False,
                edgecolor='orange',
                linewidth=2
            )
            ax1.add_patch(highlight2)
        
        return_elements = [point1, point2, img1, img2]
        if trajectory1:
            return_elements.append(trajectory1)
        if trajectory2:
            return_elements.append(trajectory2)
        if highlight1:
            return_elements.append(highlight1)
        if highlight2:
            return_elements.append(highlight2)
        
        return tuple(return_elements)
    
    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=min_length,
        init_func=init, blit=False, interval=1000/fps
    )
    
    # Set up the writer
    writer = animation.FFMpegWriter(fps=fps)
    
    # Save the animation
    print(f"Saving comparison animation to: {output_path}")
    ani.save(output_path, writer=writer, dpi=dpi)
    
    plt.close(fig)
    print(f"Comparison animation saved successfully!")
    
    return output_path

def create_interactive_comparison_display(
    tsne_embeddings1: np.ndarray,
    tsne_embeddings2: np.ndarray,
    original_images1: np.ndarray,
    original_images2: np.ndarray,
    output_dir: str,
    base_filename: str,
    title: str = "Interactive t-SNE Comparison",
    sample_rate: int = 1  # Sample rate for saving individual frames
):
    """
    Create an interactive display comparing two NPZ files with t-SNE points and corresponding video frames.
    Saves individual frames that can be browsed interactively.
    
    Args:
        tsne_embeddings1: Numpy array of t-SNE embeddings for first file (N x 2) or (N x 3)
        tsne_embeddings2: Numpy array of t-SNE embeddings for second file (M x 2) or (M x 3)
        original_images1: Numpy array of original video frames for first file (N x H x W x C)
        original_images2: Numpy array of original video frames for second file (M x H x W x C)
        output_dir: Directory to save the output files
        base_filename: Base filename for output files
        title: Title for the visualization
        sample_rate: Sample rate for saving individual frames (1 means save every frame)
    """
    print(f"Creating interactive comparison t-SNE display...")
    
    # Create output directory for frames
    frames_dir = os.path.join(output_dir, f"{base_filename}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Get the minimum length to ensure we don't go out of bounds
    min_length = min(len(tsne_embeddings1), len(tsne_embeddings2))
    
    # Calculate combined t-SNE plots limits with padding
    combined_embeddings = np.vstack([tsne_embeddings1, tsne_embeddings2])
    
    if tsne_embeddings1.shape[1] == 3:
        x_min, x_max = combined_embeddings[:, 0].min(), combined_embeddings[:, 0].max()
        y_min, y_max = combined_embeddings[:, 1].min(), combined_embeddings[:, 1].max()
        z_min, z_max = combined_embeddings[:, 2].min(), combined_embeddings[:, 2].max()
        
        # Add padding
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        z_padding = (z_max - z_min) * 0.1
        
        x_limits = (x_min - x_padding, x_max + x_padding)
        y_limits = (y_min - y_padding, y_max + y_padding)
        z_limits = (z_min - z_padding, z_max + z_padding)
    else:
        x_min, x_max = combined_embeddings[:, 0].min(), combined_embeddings[:, 0].max()
        y_min, y_max = combined_embeddings[:, 1].min(), combined_embeddings[:, 1].max()
        
        # Add padding
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        
        x_limits = (x_min - x_padding, x_max + x_padding)
        y_limits = (y_min - y_padding, y_max + y_padding)
    
    # Create the static comparison t-SNE plot with all points
    plt.figure(figsize=(10, 8))
    
    if tsne_embeddings1.shape[1] == 3:
        ax = plt.subplot(111, projection='3d')
        
        # Plot points from first file
        ax.scatter(
            tsne_embeddings1[:, 0],
            tsne_embeddings1[:, 1],
            tsne_embeddings1[:, 2],
            color='blue',
            alpha=0.5,
            s=20,
            label='File 1'
        )
        
        # Plot points from second file
        ax.scatter(
            tsne_embeddings2[:, 0],
            tsne_embeddings2[:, 1],
            tsne_embeddings2[:, 2],
            color='orange',
            alpha=0.5,
            s=20,
            label='File 2'
        )
        
        # Draw complete trajectories
        ax.plot(
            tsne_embeddings1[:, 0],
            tsne_embeddings1[:, 1],
            tsne_embeddings1[:, 2],
            'b-',
            alpha=0.8,
            linewidth=1.5,
            label='Trajectory 1'
        )
        
        ax.plot(
            tsne_embeddings2[:, 0],
            tsne_embeddings2[:, 1],
            tsne_embeddings2[:, 2],
            color='orange',
            alpha=0.8,
            linewidth=1.5,
            label='Trajectory 2'
        )
        
        # Set consistent limits
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_zlim(z_limits)
        
        # Set labels
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        ax.set_zlabel("t-SNE Component 3")
    else:
        ax = plt.subplot(111)
        
        # Plot points from first file
        ax.scatter(
            tsne_embeddings1[:, 0],
            tsne_embeddings1[:, 1],
            color='blue',
            alpha=0.5,
            s=20,
            label='File 1'
        )
        
        # Plot points from second file
        ax.scatter(
            tsne_embeddings2[:, 0],
            tsne_embeddings2[:, 1],
            color='orange',
            alpha=0.5,
            s=20,
            label='File 2'
        )
        
        # Draw complete trajectories
        ax.plot(
            tsne_embeddings1[:, 0],
            tsne_embeddings1[:, 1],
            'b-',
            alpha=0.8,
            linewidth=1.5,
            label='Trajectory 1'
        )
        
        ax.plot(
            tsne_embeddings2[:, 0],
            tsne_embeddings2[:, 1],
            color='orange',
            alpha=0.8,
            linewidth=1.5,
            label='Trajectory 2'
        )
        
        # Set consistent limits
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        
        # Set labels
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
    
    # Highlight start and end points
    for embeddings, color, label_prefix in [
        (tsne_embeddings1, 'blue', '1'),
        (tsne_embeddings2, 'orange', '2')
    ]:
        if tsne_embeddings1.shape[1] == 3:
            # Start point
            ax.scatter(
                embeddings[0, 0],
                embeddings[0, 1],
                embeddings[0, 2],
                color=color,
                s=100,
                edgecolors='black',
                marker='^',
                label=f'Start {label_prefix}'
            )
            
            # End point
            ax.scatter(
                embeddings[-1, 0],
                embeddings[-1, 1],
                embeddings[-1, 2],
                color=color,
                s=100,
                edgecolors='black',
                marker='s',
                label=f'End {label_prefix}'
            )
        else:
            # Start point
            ax.scatter(
                embeddings[0, 0],
                embeddings[0, 1],
                color=color,
                s=100,
                edgecolors='black',
                marker='^',
                label=f'Start {label_prefix}'
            )
            
            # End point
            ax.scatter(
                embeddings[-1, 0],
                embeddings[-1, 1],
                color=color,
                s=100,
                edgecolors='black',
                marker='s',
                label=f'End {label_prefix}'
            )
    
    plt.title(f"{title} - All Points")
    
    # Add legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')
    
    # Save the static plot
    static_plot_path = os.path.join(output_dir, f"{base_filename}_static.png")
    plt.savefig(static_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create HTML content for interactive visualization
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ display: flex; flex-wrap: wrap; }}
            .tsne-container {{ flex: 2; min-width: 500px; }}
            .frames-container {{ flex: 1; min-width: 400px; display: flex; flex-direction: column; }}
            .frame-box {{ margin-bottom: 20px; }}
            .controls {{ margin: 20px 0; }}
            img {{ max-width: 100%; border: 1px solid #ddd; }}
            #frameSlider {{ width: 80%; }}
            #frameDisplay {{ font-weight: bold; margin-left: 10px; }}
            .play-button {{ padding: 8px 16px; margin-right: 10px; cursor: pointer; }}
            .legend {{ margin-top: 20px; padding: 10px; border: 1px solid #ddd; background: #f9f9f9; }}
            .legend-item {{ display: flex; align-items: center; margin: 5px 0; }}
            .legend-color {{ width: 20px; height: 20px; margin-right: 10px; }}
            .blue {{ background-color: blue; }}
            .orange {{ background-color: orange; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        
        <div class="controls">
            <button id="playButton" class="play-button">Play</button>
            <input type="range" id="frameSlider" min="0" max="{min_length-1}" value="0">
            <span id="frameDisplay">Frame: 1/{min_length}</span>
        </div>
        
        <div class="container">
            <div class="tsne-container">
                <h2>t-SNE Embedding Space Comparison</h2>
                <img id="tsneImg" src="{base_filename}_frames/frame_0.png">
                
                <div class="legend">
                    <h3>Legend</h3>
                    <div class="legend-item">
                        <div class="legend-color blue"></div>
                        <span>File 1</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color orange"></div>
                        <span>File 2</span>
                    </div>
                </div>
            </div>
            
            <div class="frames-container">
                <div class="frame-box">
                    <h2>File 1 Video Frame</h2>
                    <img id="video1Img" src="{base_filename}_frames/video1_0.png">
                </div>
                
                <div class="frame-box">
                    <h2>File 2 Video Frame</h2>
                    <img id="video2Img" src="{base_filename}_frames/video2_0.png">
                </div>
            </div>
        </div>
        
        <script>
            const slider = document.getElementById('frameSlider');
            const frameDisplay = document.getElementById('frameDisplay');
            const tsneImg = document.getElementById('tsneImg');
            const video1Img = document.getElementById('video1Img');
            const video2Img = document.getElementById('video2Img');
            const playButton = document.getElementById('playButton');
            const maxFrame = {min_length-1};
            let isPlaying = false;
            let playInterval;
            
            // Update displays based on current frame
            function updateFrame(frame) {{
                frameDisplay.textContent = `Frame: ${{frame+1}}/${{maxFrame+1}}`;
                tsneImg.src = `{base_filename}_frames/frame_${{frame}}.png`;
                video1Img.src = `{base_filename}_frames/video1_${{frame}}.png`;
                video2Img.src = `{base_filename}_frames/video2_${{frame}}.png`;
                slider.value = frame;
            }}
            
            // Handle slider change
            slider.addEventListener('input', function() {{
                const frame = parseInt(this.value);
                updateFrame(frame);
                if (isPlaying) {{
                    stopPlayback();
                }}
            }});
            
            // Play button functionality
            playButton.addEventListener('click', function() {{
                if (isPlaying) {{
                    stopPlayback();
                }} else {{
                    startPlayback();
                }}
            }});
            
            function startPlayback() {{
                isPlaying = true;
                playButton.textContent = 'Pause';
                let currentFrame = parseInt(slider.value);
                
                playInterval = setInterval(() => {{
                    currentFrame++;
                    if (currentFrame > maxFrame) {{
                        currentFrame = 0;
                    }}
                    updateFrame(currentFrame);
                }}, 200); // Adjust speed as needed
            }}
            
            function stopPlayback() {{
                isPlaying = false;
                playButton.textContent = 'Play';
                clearInterval(playInterval);
            }}
        </script>
    </body>
    </html>
    """
    
    # Save frames at the specified sample rate
    for i in tqdm(range(0, min_length, sample_rate), desc="Saving comparison frames"):
        # Create t-SNE plot with current points highlighted
        plt.figure(figsize=(12, 10))
        
        if tsne_embeddings1.shape[1] == 3:
            ax = plt.subplot(111, projection='3d')
            
            # Plot all points with lower alpha
            for embeddings, color in [
                (tsne_embeddings1, 'blue'),
                (tsne_embeddings2, 'orange')
            ]:
                ax.scatter(
                    embeddings[:, 0],
                    embeddings[:, 1],
                    embeddings[:, 2],
                    color=color,
                    s=20,
                    alpha=0.2
                )
            
            # Set consistent limits
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
            ax.set_zlim(z_limits)
            
            # Draw trajectories up to current point
            if i > 0:
                # Trajectory for file 1
                ax.plot(
                    tsne_embeddings1[:i+1, 0],
                    tsne_embeddings1[:i+1, 1],
                    tsne_embeddings1[:i+1, 2],
                    'b-',
                    alpha=0.6,
                    linewidth=1.5
                )
                
                # Trajectory for file 2
                ax.plot(
                    tsne_embeddings2[:i+1, 0],
                    tsne_embeddings2[:i+1, 1],
                    tsne_embeddings2[:i+1, 2],
                    color='orange',
                    alpha=0.6,
                    linewidth=1.5
                )
            
            # Highlight current points
            ax.scatter(
                tsne_embeddings1[i, 0],
                tsne_embeddings1[i, 1],
                tsne_embeddings1[i, 2],
                color='blue',
                s=100,
                edgecolors='black'
            )
            
            ax.scatter(
                tsne_embeddings2[i, 0],
                tsne_embeddings2[i, 1],
                tsne_embeddings2[i, 2],
                color='orange',
                s=100,
                edgecolors='black'
            )
        else:
            ax = plt.subplot(111)
            
            # Plot all points with lower alpha
            for embeddings, color in [
                (tsne_embeddings1, 'blue'),
                (tsne_embeddings2, 'orange')
            ]:
                ax.scatter(
                    embeddings[:, 0],
                    embeddings[:, 1],
                    color=color,
                    s=20,
                    alpha=0.2
                )
            
            # Set consistent limits
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
            
            # Draw trajectories up to current point
            if i > 0:
                # Trajectory for file 1
                ax.plot(
                    tsne_embeddings1[:i+1, 0],
                    tsne_embeddings1[:i+1, 1],
                    'b-',
                    alpha=0.6,
                    linewidth=1.5
                )
                
                # Trajectory for file 2
                ax.plot(
                    tsne_embeddings2[:i+1, 0],
                    tsne_embeddings2[:i+1, 1],
                    color='orange',
                    alpha=0.6,
                    linewidth=1.5
                )
            
            # Highlight current points
            ax.scatter(
                tsne_embeddings1[i, 0],
                tsne_embeddings1[i, 1],
                color='blue',
                s=100,
                edgecolors='black'
            )
            
            ax.scatter(
                tsne_embeddings2[i, 0],
                tsne_embeddings2[i, 1],
                color='orange',
                s=100,
                edgecolors='black'
            )
        
        # Label start points if this is the first frame
        if i == 0:
            for embeddings, color, label in [
                (tsne_embeddings1, 'blue', 'Start 1'),
                (tsne_embeddings2, 'orange', 'Start 2')
            ]:
                ax.text(
                    embeddings[0, 0],
                    embeddings[0, 1],
                    *([] if embeddings.shape[1] == 2 else [embeddings[0, 2]]),
                    label,
                    fontsize=10,
                    verticalalignment='bottom'
                )
        
        # Add legend
        for color, label in [('blue', 'File 1'), ('orange', 'File 2')]:
            ax.scatter([], [], color=color, s=50, label=label)
        
        ax.legend()
        
        plt.title(f"t-SNE Comparison - Frame {i+1}/{min_length}")
        
        # Save t-SNE frame
        tsne_frame_path = os.path.join(frames_dir, f"frame_{i}.png")
        plt.savefig(tsne_frame_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save corresponding video frames
        # Process video frames from file 1
        video_frame1 = original_images1[i]
        if video_frame1.dtype == np.float32 or video_frame1.dtype == np.float64:
            if np.max(video_frame1) <= 1.0:
                video_frame1 = (video_frame1 * 255).astype(np.uint8)
        
        video1_frame_path = os.path.join(frames_dir, f"video1_{i}.png")
        cv2.imwrite(video1_frame_path, cv2.cvtColor(video_frame1, cv2.COLOR_RGB2BGR))
        
        # Process video frames from file 2
        video_frame2 = original_images2[i]
        if video_frame2.dtype == np.float32 or video_frame2.dtype == np.float64:
            if np.max(video_frame2) <= 1.0:
                video_frame2 = (video_frame2 * 255).astype(np.uint8)
        
        video2_frame_path = os.path.join(frames_dir, f"video2_{i}.png")
        cv2.imwrite(video2_frame_path, cv2.cvtColor(video_frame2, cv2.COLOR_RGB2BGR))
    
    # Save HTML file
    html_path = os.path.join(output_dir, f"{base_filename}_interactive.html")
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"Interactive comparison HTML saved to: {html_path}")
    
    return {
        'static_plot': static_plot_path,
        'frames_dir': frames_dir,
        'html_path': html_path
    }

def create_static_comparison_plot(
    tsne_embeddings1: np.ndarray,
    tsne_embeddings2: np.ndarray,
    output_path: str,
    title: str = "Comparison t-SNE Visualization",
    show_trajectories: bool = True,
    highlight_endpoints: bool = True
):
    """
    Create a static comparison plot of two t-SNE visualizations.
    
    Args:
        tsne_embeddings1: Numpy array of t-SNE embeddings for first file (N x 2) or (N x 3)
        tsne_embeddings2: Numpy array of t-SNE embeddings for second file (M x 2) or (M x 3)
        output_path: Path to save the plot
        title: Title for the plot
        show_trajectories: Whether to show the complete trajectories
        highlight_endpoints: Whether to highlight the start and end points
    """
    print(f"Creating static comparison plot...")
    
    # Set up the figure
    plt.figure(figsize=(12, 10))
    
    # Create subplot
    if tsne_embeddings1.shape[1] == 3:
        ax = plt.subplot(111, projection='3d')
    else:
        ax = plt.subplot(111)
    
    # Plot points from first file
    if tsne_embeddings1.shape[1] == 3:
        ax.scatter(
            tsne_embeddings1[:, 0],
            tsne_embeddings1[:, 1],
            tsne_embeddings1[:, 2],
            color='blue',
            alpha=0.5,
            s=20,
            label='File 1'
        )
    else:
        ax.scatter(
            tsne_embeddings1[:, 0],
            tsne_embeddings1[:, 1],
            color='blue',
            alpha=0.5,
            s=20,
            label='File 1'
        )
    
    # Plot points from second file
    if tsne_embeddings2.shape[1] == 3:
        ax.scatter(
            tsne_embeddings2[:, 0],
            tsne_embeddings2[:, 1],
            tsne_embeddings2[:, 2],
            color='orange',
            alpha=0.5,
            s=20,
            label='File 2'
        )
    else:
        ax.scatter(
            tsne_embeddings2[:, 0],
            tsne_embeddings2[:, 1],
            color='orange',
            alpha=0.5,
            s=20,
            label='File 2'
        )
    
    # Draw trajectories if requested
    if show_trajectories:
        if tsne_embeddings1.shape[1] == 3:
            ax.plot(
                tsne_embeddings1[:, 0],
                tsne_embeddings1[:, 1],
                tsne_embeddings1[:, 2],
                'b-',
                alpha=0.8,
                linewidth=1.5,
                label='Trajectory 1'
            )
            
            ax.plot(
                tsne_embeddings2[:, 0],
                tsne_embeddings2[:, 1],
                tsne_embeddings2[:, 2],
                color='orange',
                alpha=0.8,
                linewidth=1.5,
                label='Trajectory 2'
            )
        else:
            ax.plot(
                tsne_embeddings1[:, 0],
                tsne_embeddings1[:, 1],
                'b-',
                alpha=0.8,
                linewidth=1.5,
                label='Trajectory 1'
            )
            
            ax.plot(
                tsne_embeddings2[:, 0],
                tsne_embeddings2[:, 1],
                color='orange',
                alpha=0.8,
                linewidth=1.5,
                label='Trajectory 2'
            )
    
    # Highlight endpoints if requested
    if highlight_endpoints:
        # Highlight start and end points for first file
        if tsne_embeddings1.shape[1] == 3:
            ax.scatter(
                tsne_embeddings1[0, 0],
                tsne_embeddings1[0, 1],
                tsne_embeddings1[0, 2],
                color='blue',
                s=100,
                edgecolors='black',
                marker='^',
                label='Start 1'
            )
            
            ax.scatter(
                tsne_embeddings1[-1, 0],
                tsne_embeddings1[-1, 1],
                tsne_embeddings1[-1, 2],
                color='blue',
                s=100,
                edgecolors='black',
                marker='s',
                label='End 1'
            )
        else:
            ax.scatter(
                tsne_embeddings1[0, 0],
                tsne_embeddings1[0, 1],
                color='blue',
                s=100,
                edgecolors='black',
                marker='^',
                label='Start 1'
            )
            
            ax.scatter(
                tsne_embeddings1[-1, 0],
                tsne_embeddings1[-1, 1],
                color='blue',
                s=100,
                edgecolors='black',
                marker='s',
                label='End 1'
            )
        
        # Highlight start and end points for second file
        if tsne_embeddings2.shape[1] == 3:
            ax.scatter(
                tsne_embeddings2[0, 0],
                tsne_embeddings2[0, 1],
                tsne_embeddings2[0, 2],
                color='orange',
                s=100,
                edgecolors='black',
                marker='^',
                label='Start 2'
            )
            
            ax.scatter(
                tsne_embeddings2[-1, 0],
                tsne_embeddings2[-1, 1],
                tsne_embeddings2[-1, 2],
                color='orange',
                s=100,
                edgecolors='black',
                marker='s',
                label='End 2'
            )
        else:
            ax.scatter(
                tsne_embeddings2[0, 0],
                tsne_embeddings2[0, 1],
                color='orange',
                s=100,
                edgecolors='black',
                marker='^',
                label='Start 2'
            )
            
            ax.scatter(
                tsne_embeddings2[-1, 0],
                tsne_embeddings2[-1, 1],
                color='orange',
                s=100,
                edgecolors='black',
                marker='s',
                label='End 2'
            )
    
    # Add labels and title
    if tsne_embeddings1.shape[1] == 3:
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        ax.set_zlabel("t-SNE Component 3")
    else:
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
    
    plt.title(title)
    
    # Add legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Static comparison plot saved to: {output_path}")
    
    return output_path

def compare_npz_files(
    npz_path1: str,
    npz_path2: str,
    output_dir: str,
    policy,
    dataset,
    device: str,
    image_key: str = 'zed_sim_images',
    tsne_components: int = 2,
    tsne_perplexity: int = 30,
    tsne_iterations: int = 1000,
    create_animation: bool = True,
    create_interactive: bool = True,
    animation_fps: int = 10,
    sample_rate: int = 1
):
    """
    Compare two NPZ files by visualizing them on the same t-SNE plot.
    
    Args:
        npz_path1: Path to the first NPZ file
        npz_path2: Path to the second NPZ file
        output_dir: Directory to save processed files and visualizations
        policy: The policy model containing the encoder
        dataset: The dataset object for processing observations
        device: Device to use for processing
        image_key: Key for the images in the NPZ files
        tsne_components: Number of components for t-SNE (2 or 3)
        tsne_perplexity: Perplexity parameter for t-SNE
        tsne_iterations: Number of iterations for t-SNE
        create_animation: Whether to create animated visualizations
        animation_fps: Frames per second for animations
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create visualization directory
    vis_output_path = output_path / "visualizations"
    vis_output_path.mkdir(parents=True, exist_ok=True)
    
    # Process the two NPZ files
    data1, embeddings1, original_images1, data2, embeddings2, original_images2 = process_npz_files_for_comparison(
        npz_path1, npz_path2, policy, dataset, device, image_key
    )
    
    # Compute combined t-SNE
    tsne1, tsne2, combined_tsne = compute_combined_tsne(
        embeddings1, 
        embeddings2,
        n_components=tsne_components,
        perplexity=tsne_perplexity,
        n_iter=tsne_iterations
    )
    
    # Create base filenames for outputs
    file1_name = Path(npz_path1).stem
    file2_name = Path(npz_path2).stem
    base_filename = f"{file1_name}_vs_{file2_name}"
    
    # Create static comparison plot
    static_plot_path = vis_output_path / f"{base_filename}_static_comparison.png"
    create_static_comparison_plot(
        tsne1,
        tsne2,
        str(static_plot_path),
        title=f"t-SNE Comparison: {file1_name} vs {file2_name}"
    )
    
    # Create animation if requested
    if create_animation:
        animation_output_path = vis_output_path / "animations"
        animation_output_path.mkdir(parents=True, exist_ok=True)
        
        animation_file = animation_output_path / f"{base_filename}_comparison.mp4"
        create_comparison_tsne_animation(
            tsne1,
            tsne2,
            original_images1,
            original_images2,
            str(animation_file),
            title=f"t-SNE Comparison: {file1_name} vs {file2_name}",
            fps=animation_fps
        )
    
    # Create interactive visualization if requested
    if create_interactive:
        interactive_output_path = vis_output_path / "interactive"
        interactive_output_path.mkdir(parents=True, exist_ok=True)
        
        create_interactive_comparison_display(
            tsne1,
            tsne2,
            original_images1,
            original_images2,
            str(interactive_output_path),
            base_filename,
            title=f"t-SNE Comparison: {file1_name} vs {file2_name}",
            sample_rate=sample_rate
        )
    
    # Save processed data with embeddings and t-SNE coordinates
    for i, (data, embeddings, tsne_coords, npz_path) in enumerate([
        (data1, embeddings1, tsne1, npz_path1),
        (data2, embeddings2, tsne2, npz_path2)
    ]):
        # Add t-SNE coordinates to each timestep
        for j, timestep in enumerate(data):
            timestep['tsne_coords'] = tsne_coords[j]
        
        # Save the processed data
        output_file = output_path / f"{Path(npz_path).stem}_with_embeddings_comparison.npz"
        save_embeddings(str(output_file), {"episode": data})
    
    print(f"Comparison complete! Results saved to {output_dir}")
    
    return {
        'static_plot': str(static_plot_path),
        'animation': str(animation_file) if create_animation else None,
        'tsne1': tsne1,
        'tsne2': tsne2
    }

def find_matching_demos(dir1: str, dir2: str, match_pattern: Optional[str] = None) -> List[Tuple[str, str]]:
    """
    Find pairs of matching demos between two directories.
    
    Args:
        dir1: First directory containing NPZ files
        dir2: Second directory containing NPZ files
        match_pattern: Optional regex pattern to use for matching files.
                      If None, will match by filename (excluding extension)
    
    Returns:
        List of tuples with (path_in_dir1, path_in_dir2) for matching demos
    """
    dir1_path = Path(dir1)
    dir2_path = Path(dir2)
    
    # Get all NPZ files in both directories
    files1 = sorted([f for f in dir1_path.glob('*.npz')])
    files2 = sorted([f for f in dir2_path.glob('*.npz')])
    
    # Dictionary to store matches
    matches = []
    
    if match_pattern:
        # Use regex pattern for matching
        pattern = re.compile(match_pattern)
        
        # Create mapping from matched pattern to file path
        files1_dict = {}
        for file1 in files1:
            match = pattern.search(file1.name)
            if match:
                key = match.group(0)
                files1_dict[key] = file1
        
        # Find matches in dir2
        for file2 in files2:
            match = pattern.search(file2.name)
            if match:
                key = match.group(0)
                if key in files1_dict:
                    matches.append((str(files1_dict[key]), str(file2)))
    else:
        # Match by filename (excluding extension)
        files1_dict = {f.stem: f for f in files1}
        
        for file2 in files2:
            if file2.stem in files1_dict:
                matches.append((str(files1_dict[file2.stem]), str(file2)))
    
    return matches

def process_directory_pairs(
    dir1: str,
    dir2: str,
    output_dir: str,
    policy,
    dataset,
    device: str,
    image_key: str = 'zed_sim_images',
    match_pattern: Optional[str] = None,
    tsne_components: int = 2,
    tsne_perplexity: int = 30,
    tsne_iterations: int = 1000,
    create_animation: bool = True,
    create_interactive: bool = True,
    create_multi_demo_interactive: bool = True,
    animation_fps: int = 10,
    sample_rate: int = 1
):
    """
    Process pairs of matching demos from two directories.
    
    Args:
        dir1: First directory containing NPZ files
        dir2: Second directory containing NPZ files
        output_dir: Directory to save processed files and visualizations
        policy: The policy model containing the encoder
        dataset: The dataset object for processing observations
        device: Device to use for processing
        image_key: Key for the images in the NPZ files
        match_pattern: Optional regex pattern to use for matching files
        tsne_components: Number of components for t-SNE (2 or 3)
        tsne_perplexity: Perplexity parameter for t-SNE
        tsne_iterations: Number of iterations for t-SNE
        create_animation: Whether to create animated visualizations
        create_interactive: Whether to create interactive HTML visualizations
        create_multi_demo_interactive: Whether to create a multi-demo interactive visualization
        animation_fps: Frames per second for animations
        sample_rate: Sample rate for interactive visualization frames
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find matching demo pairs
    matches = find_matching_demos(dir1, dir2, match_pattern)
    
    if not matches:
        print(f"Error: No matching demo pairs found between {dir1} and {dir2}")
        return
    
    print(f"Found {len(matches)} matching demo pairs")
    
    # Process each pair of demos
    all_results = []
    for i, (file1, file2) in enumerate(tqdm(matches, desc="Processing demo pairs")):
        pair_name = f"pair_{i+1}_{Path(file1).stem}_vs_{Path(file2).stem}"
        pair_output_dir = output_path / pair_name
        pair_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing pair {i+1}/{len(matches)}: {Path(file1).name} vs {Path(file2).name}")
        
        try:
            result = compare_npz_files(
                file1,
                file2,
                str(pair_output_dir),
                policy,
                dataset,
                device,
                image_key,
                tsne_components=tsne_components,
                tsne_perplexity=tsne_perplexity,
                tsne_iterations=tsne_iterations,
                create_animation=create_animation,
                create_interactive=create_interactive,
                animation_fps=animation_fps,
                sample_rate=sample_rate,
                pair_id=i + 1  # Pass the pair ID
            )
            
            # Add metadata to results
            result['pair_id'] = i + 1
            result['file1'] = file1
            result['file2'] = file2
            result['pair_name'] = pair_name
            result['output_dir'] = str(pair_output_dir)
            
            all_results.append(result)
            
        except Exception as e:
            print(f"Error processing pair {file1} vs {file2}: {e}")
    
    # Create a metadata file with information about all pairs
    metadata = {
        'total_pairs': len(matches),
        'pairs': [
            {
                'pair_id': result['pair_id'],
                'file1': result['file1'],
                'file2': result['file2'],
                'pair_name': result['pair_name'],
                'output_dir': result['output_dir'],
                'static_plot': result['static_plot'],
                'animation': result['animation']
            }
            for result in all_results
        ]
    }
    
    with open(output_path / 'comparison_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create a multi-demo interactive visualization if requested
    if create_multi_demo_interactive and all_results:
        create_multi_demo_interactive_display(all_results, str(output_path))
    
    print(f"Directory pair comparison complete! Results saved to {output_dir}")
    
    return all_results

def create_multi_demo_interactive_display(results, output_dir: str):
    """
    Create an interactive HTML dashboard that allows switching between different demo comparisons.
    
    Args:
        results: List of results from compare_npz_files
        output_dir: Directory to save the HTML file
    """
    print("Creating multi-demo interactive dashboard...")
    
    output_path = Path(output_dir)
    dashboard_path = output_path / "dashboard.html"
    
    # Generate HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Multi-Demo Comparison Dashboard</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 15px;
                background-color: #f5f5f5;
                font-size: 14px;
            }
            .header {
                background-color: #2c3e50;
                color: white;
                padding: 10px 15px;
                border-radius: 5px;
                margin-bottom: 15px;
            }
            .header h1 {
                margin: 0;
                font-size: 20px;
            }
            .container {
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
            }
            .sidebar {
                flex: 1;
                min-width: 220px;
                max-width: 260px;
                background-color: white;
                border-radius: 5px;
                padding: 12px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .content {
                flex: 3;
                min-width: 500px;
                background-color: white;
                border-radius: 5px;
                padding: 12px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .demo-list {
                margin-bottom: 15px;
                max-height: 250px;
                overflow-y: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 8px;
            }
            .demo-item {
                padding: 6px 10px;
                margin: 4px 0;
                cursor: pointer;
                border-radius: 3px;
                font-size: 13px;
            }
            .demo-item:hover {
                background-color: #f0f0f0;
            }
            .demo-item.active {
                background-color: #3498db;
                color: white;
            }
            .iframe-container {
                width: 100%;
                height: 600px;
                border: 1px solid #ddd;
                border-radius: 5px;
                overflow: hidden;
            }
            iframe {
                width: 100%;
                height: 100%;
                border: none;
            }
            .options {
                margin-bottom: 15px;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .options h3 {
                margin-top: 0;
                margin-bottom: 8px;
                font-size: 15px;
            }
            .button-group {
                display: flex;
                gap: 5px;
            }
            button {
                padding: 6px 10px;
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 3px;
                cursor: pointer;
                font-size: 13px;
                flex: 1;
            }
            button:hover {
                background-color: #2980b9;
            }
            button:disabled {
                background-color: #95a5a6;
                cursor: not-allowed;
            }
            .demo-info {
                font-size: 13px;
                color: #666;
                margin-bottom: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 8px;
            }
            h2 {
                font-size: 16px;
                margin: 0 0 10px 0;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Multi-Demo Comparison Dashboard</h1>
        </div>
        <div class="container">
            <div class="sidebar">
                <h2>Demo Pairs</h2>
                <div class="demo-list">
                    <div id="demoList">
                        <!-- Demo items will be inserted here -->
                    </div>
                </div>
                <div class="options">
                    <h3>View Options</h3>
                    <div class="button-group">
                        <button id="staticBtn">Static</button>
                        <button id="animationBtn">Animation</button>
                        <button id="interactiveBtn" disabled>Interactive</button>
                    </div>
                </div>
                <div class="demo-info" id="demoInfo">
                    <!-- Demo info will be shown here -->
                </div>
            </div>
            <div class="content">
                <div class="iframe-container">
                    <iframe id="contentFrame" src="" frameborder="0"></iframe>
                </div>
            </div>
        </div>

        
        <script>
            // Demo pair data
            const demoPairs = [
    """
    
    # Add demo pair data as JavaScript objects
    for result in results:
        # Get relative paths to make links work
        static_plot_rel = os.path.relpath(result['static_plot'], output_dir)
        animation_rel = os.path.relpath(result['animation'], output_dir) if result['animation'] else ""
        
        # Get interactive HTML path if it exists
        interactive_dir = os.path.join(result['output_dir'], "visualizations", "interactive")
        interactive_html = None
        if os.path.exists(interactive_dir):
            html_files = [f for f in os.listdir(interactive_dir) if f.endswith('_interactive.html')]
            if html_files:
                interactive_html = os.path.join(interactive_dir, html_files[0])
                interactive_html = os.path.relpath(interactive_html, output_dir)
        
        # Process paths for JavaScript by replacing backslashes with forward slashes
        static_plot_js = static_plot_rel.replace('\\', '/')
        animation_js = animation_rel.replace('\\', '/') if animation_rel else ""
        interactive_js = interactive_html.replace('\\', '/') if interactive_html else ""

        html_content += f"""
                {{
                    id: {result['pair_id']},
                    name: "{result['pair_name']}",
                    file1: "{os.path.basename(result['file1'])}",
                    file2: "{os.path.basename(result['file2'])}",
                    staticPlot: "{static_plot_js}",
                    animation: "{animation_js}",
                    interactive: "{interactive_js}",
                }},"""
    
    # Add the rest of the JavaScript after the demo pairs array
    html_content += """
            ];
            
            // Current selected demo
            let currentDemo = demoPairs[0];
            
            // Populate demo list
            function populateDemoList() {
                const demoList = document.getElementById('demoList');
                demoPairs.forEach(demo => {
                    const item = document.createElement('div');
                    item.className = 'demo-item';
                    item.textContent = `Pair ${demo.id}: ${demo.file1} vs ${demo.file2}`;
                    item.onclick = () => selectDemo(demo);
                    
                    if (demo.id === currentDemo.id) {
                        item.classList.add('active');
                    }
                    
                    demoList.appendChild(item);
                });
                
                updateDemoInfo();
            }
            
            // Select a demo
            function selectDemo(demo) {
                currentDemo = demo;
                
                // Update active class
                const items = document.querySelectorAll('.demo-item');
                items.forEach(item => {
                    item.classList.remove('active');
                    if (item.textContent.includes(`Pair ${demo.id}:`)) {
                        item.classList.add('active');
                    }
                });
                
                // Update buttons based on available views
                document.getElementById('staticBtn').disabled = !demo.staticPlot;
                document.getElementById('animationBtn').disabled = !demo.animation;
                document.getElementById('interactiveBtn').disabled = !demo.interactive;
                
                // Set the default view (static plot if available)
                if (demo.staticPlot) {
                    document.getElementById('contentFrame').src = demo.staticPlot;
                } else if (demo.animation) {
                    document.getElementById('contentFrame').src = demo.animation;
                } else if (demo.interactive) {
                    document.getElementById('contentFrame').src = demo.interactive;
                }
                
                updateDemoInfo();
            }
            
            // Update demo info
            function updateDemoInfo() {
                const infoDiv = document.getElementById('demoInfo');
                infoDiv.innerHTML = `
                    <p><strong>Selected:</strong> Pair ${currentDemo.id}</p>
                    <p><strong>File 1:</strong> ${currentDemo.file1}</p>
                    <p><strong>File 2:</strong> ${currentDemo.file2}</p>
                `;
            }
            
            // Set up button event listeners
            document.getElementById('staticBtn').addEventListener('click', () => {
                if (currentDemo.staticPlot) {
                    document.getElementById('contentFrame').src = currentDemo.staticPlot;
                }
            });
            
            document.getElementById('animationBtn').addEventListener('click', () => {
                if (currentDemo.animation) {
                    document.getElementById('contentFrame').src = currentDemo.animation;
                }
            });
            
            document.getElementById('interactiveBtn').addEventListener('click', () => {
                if (currentDemo.interactive) {
                    document.getElementById('contentFrame').src = currentDemo.interactive;
                }
            });
            
            // Initialize
            window.onload = () => {
                populateDemoList();
                
                // Set the initial view
                if (currentDemo.staticPlot) {
                    document.getElementById('contentFrame').src = currentDemo.staticPlot;
                    document.getElementById('staticBtn').disabled = false;
                } else {
                    document.getElementById('staticBtn').disabled = true;
                }
                
                document.getElementById('animationBtn').disabled = !currentDemo.animation;
                document.getElementById('interactiveBtn').disabled = !currentDemo.interactive;
            };
        </script>
    </body>
    </html>
    """
    
    # Write the HTML file
    with open(dashboard_path, 'w') as f:
        f.write(html_content)
    
    print(f"Multi-demo dashboard created at: {dashboard_path}")
    
    return str(dashboard_path)
def create_enhanced_interactive_comparison_display(
    tsne_embeddings1: np.ndarray,
    tsne_embeddings2: np.ndarray,
    original_images1: np.ndarray,
    original_images2: np.ndarray,
    output_dir: str,
    base_filename: str,
    pair_id: int,
    title: str = "Interactive t-SNE Comparison",
    sample_rate: int = 1
):
    """
    Enhanced version of the interactive comparison display that includes pair information
    and additional navigation controls. Shows t-SNE plot and video frames in a single row.
    
    Args:
        tsne_embeddings1: Numpy array of t-SNE embeddings for first file (N x 2) or (N x 3)
        tsne_embeddings2: Numpy array of t-SNE embeddings for second file (M x 2) or (M x 3)
        original_images1: Numpy array of original video frames for first file (N x H x W x C)
        original_images2: Numpy array of original video frames for second file (M x H x W x C)
        output_dir: Directory to save the output files
        base_filename: Base filename for output files
        pair_id: ID of the current pair for navigation purposes
        title: Title for the visualization
        sample_rate: Sample rate for saving individual frames (1 means save every frame)
    """
    print(f"Creating enhanced interactive t-SNE comparison display...")
    
    # Create output directory for frames
    frames_dir = os.path.join(output_dir, f"{base_filename}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Get the minimum length to ensure we don't go out of bounds
    min_length = min(len(tsne_embeddings1), len(tsne_embeddings2))
    
    # Calculate combined t-SNE plots limits with padding
    combined_embeddings = np.vstack([tsne_embeddings1, tsne_embeddings2])
    
    if tsne_embeddings1.shape[1] == 3:
        x_min, x_max = combined_embeddings[:, 0].min(), combined_embeddings[:, 0].max()
        y_min, y_max = combined_embeddings[:, 1].min(), combined_embeddings[:, 1].max()
        z_min, z_max = combined_embeddings[:, 2].min(), combined_embeddings[:, 2].max()
        
        # Add padding
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        z_padding = (z_max - z_min) * 0.1
        
        x_limits = (x_min - x_padding, x_max + x_padding)
        y_limits = (y_min - y_padding, y_max + y_padding)
        z_limits = (z_min - z_padding, z_max + z_padding)
    else:
        x_min, x_max = combined_embeddings[:, 0].min(), combined_embeddings[:, 0].max()
        y_min, y_max = combined_embeddings[:, 1].min(), combined_embeddings[:, 1].max()
        
        # Add padding
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        
        x_limits = (x_min - x_padding, x_max + x_padding)
        y_limits = (y_min - y_padding, y_max + y_padding)
    
    # Create the static comparison t-SNE plot with all points
    plt.figure(figsize=(10, 8))
    
    if tsne_embeddings1.shape[1] == 3:
        ax = plt.subplot(111, projection='3d')
        
        # Plot points from first file
        ax.scatter(
            tsne_embeddings1[:, 0],
            tsne_embeddings1[:, 1],
            tsne_embeddings1[:, 2],
            color='blue',
            alpha=0.5,
            s=20,
            label='File 1'
        )
        
        # Plot points from second file
        ax.scatter(
            tsne_embeddings2[:, 0],
            tsne_embeddings2[:, 1],
            tsne_embeddings2[:, 2],
            color='orange',
            alpha=0.5,
            s=20,
            label='File 2'
        )
        
        # Draw complete trajectories
        ax.plot(
            tsne_embeddings1[:, 0],
            tsne_embeddings1[:, 1],
            tsne_embeddings1[:, 2],
            'b-',
            alpha=0.8,
            linewidth=1.5,
            label='Trajectory 1'
        )
        
        ax.plot(
            tsne_embeddings2[:, 0],
            tsne_embeddings2[:, 1],
            tsne_embeddings2[:, 2],
            color='orange',
            alpha=0.8,
            linewidth=1.5,
            label='Trajectory 2'
        )
        
        # Set consistent limits
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_zlim(z_limits)
        
        # Set labels
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        ax.set_zlabel("t-SNE Component 3")
    else:
        ax = plt.subplot(111)
        
        # Plot points from first file
        ax.scatter(
            tsne_embeddings1[:, 0],
            tsne_embeddings1[:, 1],
            color='blue',
            alpha=0.5,
            s=20,
            label='File 1'
        )
        
        # Plot points from second file
        ax.scatter(
            tsne_embeddings2[:, 0],
            tsne_embeddings2[:, 1],
            color='orange',
            alpha=0.5,
            s=20,
            label='File 2'
        )
        
        # Draw complete trajectories
        ax.plot(
            tsne_embeddings1[:, 0],
            tsne_embeddings1[:, 1],
            'b-',
            alpha=0.8,
            linewidth=1.5,
            label='Trajectory 1'
        )
        
        ax.plot(
            tsne_embeddings2[:, 0],
            tsne_embeddings2[:, 1],
            color='orange',
            alpha=0.8,
            linewidth=1.5,
            label='Trajectory 2'
        )
        
        # Set consistent limits
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        
        # Set labels
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
    
    # Highlight start and end points
    for embeddings, color, label_prefix in [
        (tsne_embeddings1, 'blue', '1'),
        (tsne_embeddings2, 'orange', '2')
    ]:
        if tsne_embeddings1.shape[1] == 3:
            # Start point
            ax.scatter(
                embeddings[0, 0],
                embeddings[0, 1],
                embeddings[0, 2],
                color=color,
                s=100,
                edgecolors='black',
                marker='^',
                label=f'Start {label_prefix}'
            )
            
            # End point
            ax.scatter(
                embeddings[-1, 0],
                embeddings[-1, 1],
                embeddings[-1, 2],
                color=color,
                s=100,
                edgecolors='black',
                marker='s',
                label=f'End {label_prefix}'
            )
        else:
            # Start point
            ax.scatter(
                embeddings[0, 0],
                embeddings[0, 1],
                color=color,
                s=100,
                edgecolors='black',
                marker='^',
                label=f'Start {label_prefix}'
            )
            
            # End point
            ax.scatter(
                embeddings[-1, 0],
                embeddings[-1, 1],
                color=color,
                s=100,
                edgecolors='black',
                marker='s',
                label=f'End {label_prefix}'
            )
    
    plt.title(f"{title} - All Points")
    
    # Add legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')
    
    # Save the static plot
    static_plot_path = os.path.join(output_dir, f"{base_filename}_static.png")
    plt.savefig(static_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create HTML content for enhanced interactive visualization
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 15px;
                background-color: #f5f5f5;
                font-size: 14px;
            }}
            .header {{
                background-color: #2c3e50;
                color: white;
                padding: 8px 15px;
                border-radius: 5px;
                margin-bottom: 10px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 20px;
            }}
            .nav-buttons {{
                display: flex;
                gap: 10px;
            }}
            .container {{
                display: flex;
                flex-direction: column;
                gap: 10px;
            }}
            .display-container {{
                display: flex;
                flex-direction: row;
                gap: 10px;
                justify-content: space-between;
                background-color: white;
                border-radius: 5px;
                padding: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .tsne-box {{
                flex: 1; /* Equal sizing with other containers */
                padding: 5px;
                border: 1px solid #eee;
                border-radius: 5px;
                min-width: 33%;
            }}
            .frames-container {{
                flex: 2;
                display: flex;
                flex-direction: row; /* Horizontal layout for video frames */
                gap: 10px;
            }}
            .frame-box {{
                flex: 1; /* Equal sizing for both video frames */
                padding: 5px;
                border: 1px solid #eee;
                border-radius: 5px;
                max-width: 50%;
            }}
            .controls {{
                margin: 10px 0;
                background-color: white;
                border-radius: 5px;
                padding: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .control-row {{
                display: flex;
                align-items: center;
                margin-bottom: 8px;
            }}
            .control-row:last-child {{
                margin-bottom: 0;
            }}
            img {{
                max-width: 100%;
                max-height: 250px; /* Reduced height to fit better in a row */
                border: 1px solid #ddd;
                border-radius: 3px;
                object-fit: contain;
                display: block;
                margin: 0 auto;
            }}
            #frameSlider {{
                flex: 1;
                margin-right: 10px;
            }}
            #frameDisplay {{
                font-weight: bold;
                min-width: 80px;
            }}
            .play-button {{
                padding: 6px 12px;
                margin-right: 10px;
                cursor: pointer;
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 3px;
            }}
            .play-button:hover {{
                background-color: #2980b9;
            }}
            .legend {{
                margin-top: 10px;
                padding: 8px;
                border: 1px solid #ddd;
                background: #f9f9f9;
                border-radius: 5px;
                display: flex;
                gap: 15px;
                justify-content: center;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
            }}
            .legend-color {{
                width: 15px;
                height: 15px;
                margin-right: 5px;
                border-radius: 50%;
            }}
            .blue {{
                background-color: blue;
            }}
            .orange {{
                background-color: orange;
            }}
            .speed-control {{
                display: flex;
                align-items: center;
            }}
            .speed-label {{
                margin-right: 5px;
            }}
            .nav-button {{
                padding: 6px 12px;
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 3px;
                cursor: pointer;
                text-decoration: none;
                font-size: 13px;
            }}
            .nav-button:hover {{
                background-color: #2980b9;
            }}
            .pair-info {{
                margin: 5px 0;
                font-size: 13px;
            }}
            h2 {{
                font-size: 16px;
                margin: 5px 0 8px 0;
                text-align: center;
            }}
            h3 {{
                font-size: 14px;
                margin: 0;
                display: none;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{title}</h1>
            <div class="nav-buttons">
                <a href="../dashboard.html" class="nav-button">Back to Dashboard</a>
            </div>
        </div>
        
        <div class="pair-info">
            <p><strong>Current Pair:</strong> {pair_id}</p>
        </div>
        
        <div class="controls">
            <div class="control-row">
                <button id="playButton" class="play-button">Play</button>
                <div class="speed-control">
                    <label class="speed-label">Speed:</label>
                    <select id="speedSelect">
                        <option value="500">0.5x</option>
                        <option value="200" selected>1x</option>
                        <option value="100">2x</option>
                        <option value="50">4x</option>
                    </select>
                </div>
                <span id="frameDisplay">Frame: 1/{min_length}</span>
            </div>
            <div class="control-row">
                <input type="range" id="frameSlider" min="0" max="{min_length-1}" value="0">
            </div>
        </div>
        
        <div class="container">
            <div class="display-container">
                <div class="tsne-box">
                    <h2>t-SNE Embedding Space Comparison</h2>
                    <img id="tsneImg" src="{base_filename}_frames/frame_0.png">
                    
                    <div class="legend">
                        <div class="legend-item">
                            <div class="legend-color blue"></div>
                            <span>File 1</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color orange"></div>
                            <span>File 2</span>
                        </div>
                    </div>
                </div>
                
                <div class="frames-container">
                    <div class="frame-box">
                        <h2>File 1 Video Frame</h2>
                        <img id="video1Img" src="{base_filename}_frames/video1_0.png">
                    </div>
                    
                    <div class="frame-box">
                        <h2>File 2 Video Frame</h2>
                        <img id="video2Img" src="{base_filename}_frames/video2_0.png">
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            const slider = document.getElementById('frameSlider');
            const frameDisplay = document.getElementById('frameDisplay');
            const tsneImg = document.getElementById('tsneImg');
            const video1Img = document.getElementById('video1Img');
            const video2Img = document.getElementById('video2Img');
            const playButton = document.getElementById('playButton');
            const speedSelect = document.getElementById('speedSelect');
            const maxFrame = {min_length-1};
            let isPlaying = false;
            let playInterval;
            
            // Update displays based on current frame
            function updateFrame(frame) {{
                frameDisplay.textContent = `Frame: ${{frame+1}}/${{maxFrame+1}}`;
                tsneImg.src = `{base_filename}_frames/frame_${{frame}}.png`;
                video1Img.src = `{base_filename}_frames/video1_${{frame}}.png`;
                video2Img.src = `{base_filename}_frames/video2_${{frame}}.png`;
                slider.value = frame;
            }}
            
            // Handle slider change
            slider.addEventListener('input', function() {{
                const frame = parseInt(this.value);
                updateFrame(frame);
                if (isPlaying) {{
                    stopPlayback();
                }}
            }});
            
            // Play button functionality
            playButton.addEventListener('click', function() {{
                if (isPlaying) {{
                    stopPlayback();
                }} else {{
                    startPlayback();
                }}
            }});
            
            // Speed select functionality
            speedSelect.addEventListener('change', function() {{
                if (isPlaying) {{
                    stopPlayback();
                    startPlayback();
                }}
            }});
            
            function startPlayback() {{
                isPlaying = true;
                playButton.textContent = 'Pause';
                let currentFrame = parseInt(slider.value);
                
                playInterval = setInterval(() => {{
                    currentFrame++;
                    if (currentFrame > maxFrame) {{
                        currentFrame = 0;
                    }}
                    updateFrame(currentFrame);
                }}, parseInt(speedSelect.value)); // Use the selected speed
            }}
            
            function stopPlayback() {{
                isPlaying = false;
                playButton.textContent = 'Play';
                clearInterval(playInterval);
            }}
            
            // Keyboard shortcuts
            document.addEventListener('keydown', function(e) {{
                // Space bar toggles play/pause
                if (e.code === 'Space') {{
                    e.preventDefault();
                    if (isPlaying) {{
                        stopPlayback();
                    }} else {{
                        startPlayback();
                    }}
                }}
                
                // Arrow keys for frame navigation
                let currentFrame = parseInt(slider.value);
                
                if (e.code === 'ArrowRight') {{
                    e.preventDefault();
                    currentFrame = Math.min(currentFrame + 1, maxFrame);
                    updateFrame(currentFrame);
                }}
                
                if (e.code === 'ArrowLeft') {{
                    e.preventDefault();
                    currentFrame = Math.max(currentFrame - 1, 0);
                    updateFrame(currentFrame);
                }}
                
                // Page Up/Down for bigger jumps
                if (e.code === 'PageUp') {{
                    e.preventDefault();
                    currentFrame = Math.max(currentFrame - 10, 0);
                    updateFrame(currentFrame);
                }}
                
                if (e.code === 'PageDown') {{
                    e.preventDefault();
                    currentFrame = Math.min(currentFrame + 10, maxFrame);
                    updateFrame(currentFrame);
                }}
                
                // Home/End for first/last frame
                if (e.code === 'Home') {{
                    e.preventDefault();
                    currentFrame = 0;
                    updateFrame(currentFrame);
                }}
                
                if (e.code === 'End') {{
                    e.preventDefault();
                    currentFrame = maxFrame;
                    updateFrame(currentFrame);
                }}
            }});
            
            // Initialize on page load
            window.onload = function() {{
                updateFrame(0);
            }};
        </script>
    </body>
    </html>
    """
    
    # Write the HTML file
    html_path = os.path.join(output_dir, f"{base_filename}_interactive.html")
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    # Save frames at the specified sample rate
    for i in tqdm(range(0, min_length, sample_rate), desc="Saving comparison frames"):
        # Create t-SNE plot with current points highlighted
        plt.figure(figsize=(12, 10))
        
        if tsne_embeddings1.shape[1] == 3:
            ax = plt.subplot(111, projection='3d')
            
            # Plot all points with lower alpha
            for embeddings, color in [
                (tsne_embeddings1, 'blue'),
                (tsne_embeddings2, 'orange')
            ]:
                ax.scatter(
                    embeddings[:, 0],
                    embeddings[:, 1],
                    embeddings[:, 2],
                    color=color,
                    s=20,
                    alpha=0.2
                )
            
            # Set consistent limits
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
            ax.set_zlim(z_limits)
            
            # Draw trajectories up to current point
            if i > 0:
                # Trajectory for file 1
                ax.plot(
                    tsne_embeddings1[:i+1, 0],
                    tsne_embeddings1[:i+1, 1],
                    tsne_embeddings1[:i+1, 2],
                    'b-',
                    alpha=0.6,
                    linewidth=1.5
                )
                
                # Trajectory for file 2
                ax.plot(
                    tsne_embeddings2[:i+1, 0],
                    tsne_embeddings2[:i+1, 1],
                    tsne_embeddings2[:i+1, 2],
                    color='orange',
                    alpha=0.6,
                    linewidth=1.5
                )
            
            # Highlight current points
            ax.scatter(
                tsne_embeddings1[i, 0],
                tsne_embeddings1[i, 1],
                tsne_embeddings1[i, 2],
                color='blue',
                s=100,
                edgecolors='black'
            )
            
            ax.scatter(
                tsne_embeddings2[i, 0],
                tsne_embeddings2[i, 1],
                tsne_embeddings2[i, 2],
                color='orange',
                s=100,
                edgecolors='black'
            )
        else:
            ax = plt.subplot(111)
            
            # Plot all points with lower alpha
            for embeddings, color in [
                (tsne_embeddings1, 'blue'),
                (tsne_embeddings2, 'orange')
            ]:
                ax.scatter(
                    embeddings[:, 0],
                    embeddings[:, 1],
                    color=color,
                    s=20,
                    alpha=0.2
                )
            
            # Set consistent limits
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
            
            # Draw trajectories up to current point
            if i > 0:
                # Trajectory for file 1
                ax.plot(
                    tsne_embeddings1[:i+1, 0],
                    tsne_embeddings1[:i+1, 1],
                    'b-',
                    alpha=0.6,
                    linewidth=1.5
                )
                
                # Trajectory for file 2
                ax.plot(
                    tsne_embeddings2[:i+1, 0],
                    tsne_embeddings2[:i+1, 1],
                    color='orange',
                    alpha=0.6,
                    linewidth=1.5
                )
            
            # Highlight current points
            ax.scatter(
                tsne_embeddings1[i, 0],
                tsne_embeddings1[i, 1],
                color='blue',
                s=100,
                edgecolors='black'
            )
            
            ax.scatter(
                tsne_embeddings2[i, 0],
                tsne_embeddings2[i, 1],
                color='orange',
                s=100,
                edgecolors='black'
            )
        
        # Label start points if this is the first frame
        if i == 0:
            for embeddings, color, label in [
                (tsne_embeddings1, 'blue', 'Start 1'),
                (tsne_embeddings2, 'orange', 'Start 2')
            ]:
                ax.text(
                    embeddings[0, 0],
                    embeddings[0, 1],
                    *([] if embeddings.shape[1] == 2 else [embeddings[0, 2]]),
                    label,
                    fontsize=10,
                    verticalalignment='bottom'
                )
        
        # Add legend
        for color, label in [('blue', 'File 1'), ('orange', 'File 2')]:
            ax.scatter([], [], color=color, s=50, label=label)
        
        ax.legend()
        
        plt.title(f"t-SNE Comparison - Frame {i+1}/{min_length}")
        
        # Save t-SNE frame
        tsne_frame_path = os.path.join(frames_dir, f"frame_{i}.png")
        plt.savefig(tsne_frame_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save corresponding video frames
        # Process video frames from file 1
        video_frame1 = original_images1[i]
        if video_frame1.dtype == np.float32 or video_frame1.dtype == np.float64:
            if np.max(video_frame1) <= 1.0:
                video_frame1 = (video_frame1 * 255).astype(np.uint8)
        
        video1_frame_path = os.path.join(frames_dir, f"video1_{i}.png")
        cv2.imwrite(video1_frame_path, cv2.cvtColor(video_frame1, cv2.COLOR_RGB2BGR))
        
        # Process video frames from file 2
        video_frame2 = original_images2[i]
        if video_frame2.dtype == np.float32 or video_frame2.dtype == np.float64:
            if np.max(video_frame2) <= 1.0:
                video_frame2 = (video_frame2 * 255).astype(np.uint8)
        
        video2_frame_path = os.path.join(frames_dir, f"video2_{i}.png")
        cv2.imwrite(video2_frame_path, cv2.cvtColor(video_frame2, cv2.COLOR_RGB2BGR))
    
    print(f"Interactive comparison HTML saved to: {html_path}")
    
    return {
        'static_plot': static_plot_path,
        'frames_dir': frames_dir,
        'html_path': html_path
    }

def compare_npz_files(
    npz_path1: str,
    npz_path2: str,
    output_dir: str,
    policy,
    dataset,
    device: str,
    image_key: str = 'zed_sim_images',
    tsne_components: int = 2,
    tsne_perplexity: int = 30,
    tsne_iterations: int = 1000,
    create_animation: bool = True,
    create_interactive: bool = True,
    animation_fps: int = 10,
    sample_rate: int = 1,
    pair_id: int = 1  # Added pair_id parameter for use in multi-demo comparisons
):
    """
    Compare two NPZ files by visualizing them on the same t-SNE plot.
    
    Args:
        npz_path1: Path to the first NPZ file
        npz_path2: Path to the second NPZ file
        output_dir: Directory to save processed files and visualizations
        policy: The policy model containing the encoder
        dataset: The dataset object for processing observations
        device: Device to use for processing
        image_key: Key for the images in the NPZ files
        tsne_components: Number of components for t-SNE (2 or 3)
        tsne_perplexity: Perplexity parameter for t-SNE
        tsne_iterations: Number of iterations for t-SNE
        create_animation: Whether to create animated visualizations
        animation_fps: Frames per second for animations
        pair_id: ID number for this pair when used in directory comparison
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create visualization directory
    vis_output_path = output_path / "visualizations"
    vis_output_path.mkdir(parents=True, exist_ok=True)
    
    # Process the two NPZ files
    data1, embeddings1, original_images1, data2, embeddings2, original_images2 = process_npz_files_for_comparison(
        npz_path1, npz_path2, policy, dataset, device, image_key
    )

    # normalize embeddings1 and embeddings2 with pytorch
    # embeddings1 = torch.tensor(embeddings1).to(device)
    # embeddings2 = torch.tensor(embeddings2).to(device)
    # embeddings1 = F.normalize(embeddings1, p=2, dim=1).cpu().numpy()
    # embeddings2 = F.normalize(embeddings2, p=2, dim=1).cpu().numpy()

    
    # Compute combined t-SNE
    tsne1, tsne2, combined_tsne = compute_combined_tsne(
        embeddings1, 
        embeddings2,
        n_components=tsne_components,
        perplexity=tsne_perplexity,
        n_iter=tsne_iterations
    )
    
    # Create base filenames for outputs
    file1_name = Path(npz_path1).stem
    file2_name = Path(npz_path2).stem
    base_filename = f"{file1_name}_vs_{file2_name}"
    
    # Create static comparison plot
    static_plot_path = vis_output_path / f"{base_filename}_static_comparison.png"
    create_static_comparison_plot(
        tsne1,
        tsne2,
        str(static_plot_path),
        title=f"t-SNE Comparison: {file1_name} vs {file2_name}"
    )
    
    # Create animation if requested
    animation_file = None
    if create_animation:
        animation_output_path = vis_output_path / "animations"
        animation_output_path.mkdir(parents=True, exist_ok=True)
        
        animation_file = animation_output_path / f"{base_filename}_comparison.mp4"
        create_comparison_tsne_animation(
            tsne1,
            tsne2,
            original_images1,
            original_images2,
            str(animation_file),
            title=f"t-SNE Comparison: {file1_name} vs {file2_name}",
            fps=animation_fps
        )
    
    # Create interactive visualization if requested
    if create_interactive:
        interactive_output_path = vis_output_path / "interactive"
        interactive_output_path.mkdir(parents=True, exist_ok=True)
        
        # Use the enhanced version with pair_id
        create_enhanced_interactive_comparison_display(
            tsne1,
            tsne2,
            original_images1,
            original_images2,
            str(interactive_output_path),
            base_filename,
            pair_id=pair_id,  # Pass the pair_id
            title=f"t-SNE Comparison: {file1_name} vs {file2_name}",
            sample_rate=sample_rate
        )
    
    # Save processed data with embeddings and t-SNE coordinates
    for i, (data, embeddings, tsne_coords, npz_path) in enumerate([
        (data1, embeddings1, tsne1, npz_path1),
        (data2, embeddings2, tsne2, npz_path2)
    ]):
        # Add t-SNE coordinates to each timestep
        for j, timestep in enumerate(data):
            timestep['tsne_coords'] = tsne_coords[j]
        
        # Save the processed data
        output_file = output_path / f"{Path(npz_path).stem}_with_embeddings_comparison.npz"
        save_embeddings(str(output_file), {"episode": data})
    
    print(f"Comparison complete! Results saved to {output_dir}")
    
    return {
        'static_plot': str(static_plot_path),
        'animation': str(animation_file) if animation_file else None,
        'tsne1': tsne1,
        'tsne2': tsne2
    }

def process_directory_pairs_improved(
    dir1: str,
    dir2: str,
    output_dir: str,
    policy,
    dataset,
    device: str,
    image_key: str = 'zed_sim_images',
    match_pattern: Optional[str] = None,
    tsne_components: int = 2,
    tsne_perplexity: int = 30,
    tsne_iterations: int = 1000,
    create_animation: bool = True,
    create_interactive: bool = True,
    create_multi_demo_interactive: bool = True,
    animation_fps: int = 10,
    sample_rate: int = 1
):
    """
    Process pairs of matching demos from two directories with improved t-SNE.
    Computes t-SNE on all embeddings at once, then visualizes individual pairs.
    
    Args:
        dir1: First directory containing NPZ files
        dir2: Second directory containing NPZ files
        output_dir: Directory to save processed files and visualizations
        policy: The policy model containing the encoder
        dataset: The dataset object for processing observations
        device: Device to use for processing
        image_key: Key for the images in the NPZ files
        match_pattern: Optional regex pattern to use for matching files
        tsne_components: Number of components for t-SNE (2 or 3)
        tsne_perplexity: Perplexity parameter for t-SNE
        tsne_iterations: Number of iterations for t-SNE
        create_animation: Whether to create animated visualizations
        create_interactive: Whether to create interactive HTML visualizations
        create_multi_demo_interactive: Whether to create a multi-demo interactive visualization
        animation_fps: Frames per second for animations
        sample_rate: Sample rate for interactive visualization frames
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find matching demo pairs
    matches = find_matching_demos(dir1, dir2, match_pattern)
    
    if not matches:
        print(f"Error: No matching demo pairs found between {dir1} and {dir2}")
        return
    
    print(f"Found {len(matches)} matching demo pairs")
    
    # First pass: process all files and collect embeddings
    print("First pass: processing all files and collecting embeddings...")
    all_embeddings = []
    all_file_data = []
    for i, (file1, file2) in enumerate(tqdm(matches, desc="Processing files")):
        try:
            # Process first file
            data1, embeddings1, original_images1 = process_npz_file(
                file1, policy, dataset, device, image_key
            )
            
            # Process second file
            data2, embeddings2, original_images2 = process_npz_file(
                file2, policy, dataset, device, image_key
            )
            
            
            # Store all data
            all_embeddings.append(embeddings1)
            all_embeddings.append(embeddings2)
            
            all_file_data.append({
                'pair_id': i + 1,
                'file1': file1,
                'file2': file2,
                'data1': data1,
                'data2': data2,
                'embeddings1': embeddings1,
                'embeddings2': embeddings2,
                'original_images1': original_images1,
                'original_images2': original_images2
            })

            # Compute embedding statistics
            if len(embeddings1) > 0 and len(embeddings2) > 0:
                # Random sample 20 indices (or fewer if embeddings are shorter)
                n_samples = min(20, len(embeddings1), len(embeddings2))
                random_indices = np.random.choice(min(len(embeddings1), len(embeddings2)), 
                                                 size=n_samples, replace=False)
                
                # Sample embeddings
                sampled_emb1 = embeddings1[:-1]
                sampled_emb2 = embeddings2
                
                # Compute L2 distances
                l2_distances = np.linalg.norm(sampled_emb1 - sampled_emb2, axis=1)
                l2_mean = np.mean(l2_distances)
                l2_std = np.std(l2_distances)
                
                # Compute cosine distances (1 - cosine similarity)
                # Normalize embeddings for cosine similarity
                norm_emb1 = sampled_emb1 / np.linalg.norm(sampled_emb1, axis=1, keepdims=True)
                norm_emb2 = sampled_emb2 / np.linalg.norm(sampled_emb2, axis=1, keepdims=True)
                
                cosine_similarities = np.sum(norm_emb1 * norm_emb2, axis=1)
                cosine_distances = 1 - cosine_similarities
                cosine_mean = np.mean(cosine_distances)
                cosine_std = np.std(cosine_distances)
                
                print(f"Pair {i+1} - Embedding distances (n={n_samples}):")
                print(f"  L2 distance: {l2_mean:.4f} ± {l2_std:.4f}")
                print(f"  Cosine distance: {cosine_mean:.4f} ± {cosine_std:.4f}")
                breakpoint()
            
        except Exception as e:
            print(f"Error processing pair {file1} vs {file2}: {e}")
    
    # Concatenate all embeddings for t-SNE computation
    print("Concatenating all embeddings for global t-SNE computation...")
    combined_embeddings = np.vstack(all_embeddings)
    
    # Compute t-SNE on all embeddings together
    print(f"Computing t-SNE on all {len(combined_embeddings)} embeddings...")
    tsne = TSNE(
        n_components=tsne_components, 
        perplexity=tsne_perplexity, 
        n_iter=tsne_iterations, #max_iter
        random_state=42
    )
    all_tsne_embeddings = tsne.fit_transform(combined_embeddings)
    
    # Split the t-SNE embeddings back to individual files
    print("Splitting t-SNE embeddings back to individual files...")
    embedding_start_idx = 0
    all_results = []
    
    for file_data in tqdm(all_file_data, desc="Generating visualizations"):
        # Extract range of embeddings for this pair
        emb1_length = len(file_data['embeddings1'])
        emb2_length = len(file_data['embeddings2'])
        
        # Get the corresponding t-SNE embeddings
        tsne1 = all_tsne_embeddings[embedding_start_idx:embedding_start_idx + emb1_length]
        tsne2 = all_tsne_embeddings[embedding_start_idx + emb1_length:embedding_start_idx + emb1_length + emb2_length]
        
        # Update index for next pair
        embedding_start_idx += emb1_length + emb2_length
        
        # Create output directory for this pair
        pair_name = f"pair_{file_data['pair_id']}_{Path(file_data['file1']).stem}_vs_{Path(file_data['file2']).stem}"
        pair_output_dir = output_path / pair_name
        pair_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating visualizations for pair {file_data['pair_id']}/{len(all_file_data)}: "
              f"{Path(file_data['file1']).name} vs {Path(file_data['file2']).name}")
        
        # Create visualizations
        result = generate_visualizations(
            tsne1,
            tsne2,
            file_data['original_images1'],
            file_data['original_images2'],
            file_data['data1'],
            file_data['data2'],
            file_data['file1'],
            file_data['file2'],
            str(pair_output_dir),
            pair_id=file_data['pair_id'],
            create_animation=create_animation,
            create_interactive=create_interactive,
            animation_fps=animation_fps,
            sample_rate=sample_rate
        )
        
        all_results.append(result)
    
    # Create a metadata file with information about all pairs
    metadata = {
        'total_pairs': len(matches),
        'pairs': [
            {
                'pair_id': result['pair_id'],
                'file1': result['file1'],
                'file2': result['file2'],
                'pair_name': result['pair_name'],
                'output_dir': result['output_dir'],
                'static_plot': result['static_plot'],
                'animation': result['animation']
            }
            for result in all_results
        ]
    }
    
    with open(output_path / 'comparison_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create a multi-demo interactive visualization if requested
    if create_multi_demo_interactive and all_results:
        create_multi_demo_interactive_display(all_results, str(output_path))
    
    print(f"Directory pair comparison complete! Results saved to {output_dir}")
    
    return all_results

def generate_visualizations(
    tsne1: np.ndarray,
    tsne2: np.ndarray,
    original_images1: np.ndarray,
    original_images2: np.ndarray,
    data1: Dict,
    data2: Dict,
    file1: str,
    file2: str,
    output_dir: str,
    pair_id: int = 1,
    create_animation: bool = True,
    create_interactive: bool = True,
    animation_fps: int = 10,
    sample_rate: int = 1
):
    """
    Generate visualizations for a pair of files using pre-computed t-SNE embeddings.
    
    Args:
        tsne1: t-SNE embeddings for first file
        tsne2: t-SNE embeddings for second file
        original_images1: Original images from first file
        original_images2: Original images from second file
        data1: Data from first file
        data2: Data from second file
        file1: Path to first file
        file2: Path to second file
        output_dir: Directory to save output
        pair_id: ID of the pair
        create_animation: Whether to create animated visualization
        create_interactive: Whether to create interactive visualization
        animation_fps: Frames per second for animation
        sample_rate: Sample rate for interactive visualization frames
    
    Returns:
        Dictionary with results information
    """
    output_path = Path(output_dir)
    vis_output_path = output_path / "visualizations"
    vis_output_path.mkdir(parents=True, exist_ok=True)
    
    # Create base filenames for outputs
    file1_name = Path(file1).stem
    file2_name = Path(file2).stem
    base_filename = f"{file1_name}_vs_{file2_name}"
    
    # Create static comparison plot
    static_plot_path = vis_output_path / f"{base_filename}_static_comparison.png"
    create_static_comparison_plot(
        tsne1,
        tsne2,
        str(static_plot_path),
        title=f"t-SNE Comparison: {file1_name} vs {file2_name}"
    )
    
    # Create animation if requested
    animation_file = None
    if create_animation:
        animation_output_path = vis_output_path / "animations"
        animation_output_path.mkdir(parents=True, exist_ok=True)
        
        animation_file = animation_output_path / f"{base_filename}_comparison.mp4"
        create_comparison_tsne_animation(
            tsne1,
            tsne2,
            original_images1,
            original_images2,
            str(animation_file),
            title=f"t-SNE Comparison: {file1_name} vs {file2_name}",
            fps=animation_fps
        )
    
    # Create interactive visualization if requested
    if create_interactive:
        interactive_output_path = vis_output_path / "interactive"
        interactive_output_path.mkdir(parents=True, exist_ok=True)
        
        create_enhanced_interactive_comparison_display(
            tsne1,
            tsne2,
            original_images1,
            original_images2,
            str(interactive_output_path),
            base_filename,
            pair_id=pair_id,
            title=f"t-SNE Comparison: {file1_name} vs {file2_name}",
            sample_rate=sample_rate
        )
    
    # Save processed data with embeddings and t-SNE coordinates
    for i, (data, tsne_coords, npz_path) in enumerate([
        (data1, tsne1, file1),
        (data2, tsne2, file2)
    ]):
        # Add t-SNE coordinates to each timestep
        for j, timestep in enumerate(data):
            timestep['tsne_coords'] = tsne_coords[j]
        
        # Save the processed data
        output_file = output_path / f"{Path(npz_path).stem}_with_embeddings_comparison.npz"
        save_embeddings(str(output_file), {"episode": data})
    
    return {
        'pair_id': pair_id,
        'file1': file1,
        'file2': file2,
        'pair_name': f"pair_{pair_id}_{file1_name}_vs_{file2_name}",
        'output_dir': str(output_path),
        'static_plot': str(static_plot_path),
        'animation': str(animation_file) if animation_file else None
    }


def main():
    parser = argparse.ArgumentParser(description="Extract and encode images from NPZ files with live visualization")
    parser.add_argument("--input", type=str, help="Input NPZ file or directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory for processed files")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to policy model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device for processing (cuda/cpu)")
    parser.add_argument("--image_key", type=str, default="zed_sim_images", help="Key for images in NPZ file")
    
    # t-SNE visualization arguments
    parser.add_argument("--visualize_tsne", action="store_true", help="Generate t-SNE visualizations")
    parser.add_argument("--tsne_components", type=int, default=2, choices=[2, 3], 
                        help="Number of components for t-SNE (2 or 3)")
    parser.add_argument("--tsne_perplexity", type=int, default=5, help="Perplexity parameter for t-SNE")
    parser.add_argument("--tsne_iterations", type=int, default=1000, help="Number of iterations for t-SNE")
    
    # Animation arguments
    parser.add_argument("--create_animation", action="store_true", help="Create animated visualizations")
    parser.add_argument("--animation_fps", type=int, default=2, help="Frames per second for animations")
    
    # Interactive visualization arguments
    parser.add_argument("--create_interactive", action="store_true", help="Create interactive HTML visualizations")
    parser.add_argument("--sample_rate", type=int, default=1, help="Sample rate for interactive visualization frames")
    
    # Comparison arguments
    parser.add_argument("--compare", action="store_true", help="Compare two NPZ files")
    parser.add_argument("--input1", type=str, help="First NPZ file for comparison")
    parser.add_argument("--input2", type=str, help="Second NPZ file for comparison")
    parser.add_argument("--comparison_sample_rate", type=int, default=1, 
                        help="Sample rate for interactive comparison visualization frames")
    
    # Directory comparison arguments (new)
    parser.add_argument("--compare_dirs", action="store_true", help="Compare two directories of NPZ files")
    parser.add_argument("--dir1", type=str, help="First directory containing NPZ files")
    parser.add_argument("--dir2", type=str, help="Second directory containing NPZ files")
    parser.add_argument("--match_pattern", type=str, help="Optional regex pattern for matching files between directories")
    parser.add_argument("--create_multi_demo_interactive", action="store_true", 
                        help="Create a multi-demo interactive dashboard for directory comparisons")
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # Load the policy model with encoder
    policy, dataset = setup_encoder(args.checkpoint, args.device)
    print("Successfully loaded policy model with encoder")
    
    # Check if we're doing a directory comparison
    if args.compare_dirs:
        if not args.dir1 or not args.dir2:
            print("Error: For directory comparison, both --dir1 and --dir2 must be specified")
            return
        
        # Perform directory comparison
        # process_directory_pairs(
        #     args.dir1,
        #     args.dir2,
        #     args.output,
        #     policy,
        #     dataset,
        #     args.device,
        #     args.image_key,
        #     match_pattern=args.match_pattern,
        #     tsne_components=args.tsne_components,
        #     tsne_perplexity=args.tsne_perplexity,
        #     tsne_iterations=args.tsne_iterations,
        #     create_animation=args.create_animation,
        #     create_interactive=args.create_interactive,
        #     create_multi_demo_interactive=args.create_multi_demo_interactive,
        #     animation_fps=args.animation_fps,
        #     sample_rate=args.comparison_sample_rate
        # )
        process_directory_pairs_improved(
            args.dir1,
            args.dir2,
            args.output,
            policy,
            dataset,
            args.device,
            args.image_key,
            match_pattern=args.match_pattern,
            tsne_components=args.tsne_components,
            tsne_perplexity=args.tsne_perplexity,
            tsne_iterations=args.tsne_iterations,
            create_animation=args.create_animation,
            create_interactive=args.create_interactive,
            create_multi_demo_interactive=args.create_multi_demo_interactive,
            animation_fps=args.animation_fps,
            sample_rate=args.comparison_sample_rate
        )
    # Check if we're doing a single pair comparison
    elif args.compare:
        if not args.input1 or not args.input2:
            print("Error: For comparison, both --input1 and --input2 must be specified")
            return
        
        # Perform comparison
        compare_npz_files(
            args.input1,
            args.input2,
            args.output,
            policy,
            dataset,
            args.device,
            args.image_key,
            tsne_components=args.tsne_components,
            tsne_perplexity=args.tsne_perplexity,
            tsne_iterations=args.tsne_iterations,
            create_animation=args.create_animation,
            create_interactive=args.create_interactive,
            animation_fps=args.animation_fps,
            sample_rate=args.comparison_sample_rate
        )
    # Regular processing (single file or directory)
    elif args.input:
        if os.path.isfile(args.input) and args.input.endswith('.npz'):
            # Process single file
            data, embeddings, original_images = process_npz_file(
                args.input, policy, dataset, args.device, args.image_key
            )
            output_file = os.path.join(
                args.output, 
                f"{os.path.basename(args.input).split('.')[0]}_with_embeddings.npz"
            )
            save_embeddings(output_file, {"episode": data})
            
            # Generate t-SNE visualization if requested
            if args.visualize_tsne:
                # Create visualization directory
                vis_output_dir = os.path.join(args.output, "visualizations")
                os.makedirs(vis_output_dir, exist_ok=True)
                
                # Compute t-SNE
                tsne_embeddings = compute_tsne(
                    embeddings, 
                    n_components=args.tsne_components,
                    perplexity=args.tsne_perplexity,
                    n_iter=args.tsne_iterations
                )
                
                # Create animation if requested
                if args.create_animation:
                    animation_dir = os.path.join(vis_output_dir, "animations")
                    os.makedirs(animation_dir, exist_ok=True)
                    
                    animation_file = os.path.join(
                        animation_dir, 
                        f"{os.path.basename(args.input).split('.')[0]}_tsne_animation.mp4"
                    )
                    
                    create_live_tsne_animation(
                        tsne_embeddings,
                        original_images,
                        animation_file,
                        title=f"t-SNE Visualization - {os.path.basename(args.input).split('.')[0]}",
                        fps=args.animation_fps
                    )
                
                # Create interactive visualization if requested
                if args.create_interactive:
                    interactive_dir = os.path.join(vis_output_dir, "interactive")
                    os.makedirs(interactive_dir, exist_ok=True)
                    
                    create_interactive_tsne_display(
                        tsne_embeddings,
                        original_images,
                        interactive_dir,
                        os.path.basename(args.input).split('.')[0],
                        title=f"t-SNE Visualization - {os.path.basename(args.input).split('.')[0]}",
                        sample_rate=args.sample_rate
                    )
        else:
            # Process directory
            process_directory(
                args.input, 
                args.output, 
                policy, 
                dataset,
                args.device, 
                args.image_key,
                visualize_tsne=args.visualize_tsne,
                tsne_components=args.tsne_components,
                tsne_perplexity=args.tsne_perplexity,
                tsne_iterations=args.tsne_iterations,
                create_animation=args.create_animation,
                create_interactive=args.create_interactive,
                animation_fps=args.animation_fps,
                sample_rate=args.sample_rate
            )
    else:
        print("Error: One of the following options must be specified:")
        print("  --input for processing a single file or directory")
        print("  --compare with --input1 and --input2 for comparing two files")
        print("  --compare_dirs with --dir1 and --dir2 for comparing directories of files")
        return
    
    print("Processing complete!")

if __name__ == "__main__":
    main()