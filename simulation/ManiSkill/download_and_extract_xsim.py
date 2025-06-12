#!/usr/bin/env python3
"""
Script to download xsim assets from Box and extract to the correct directory.
Usage: python download_and_extract_xsim.py <box_download_url>
"""

import os
import sys
import requests
import zipfile
import tempfile
import shutil
from pathlib import Path

# Import the proper asset directory path from mani_skill
from mani_skill import PACKAGE_ASSET_DIR

def download_file(url, filename):
    """Download file from URL with progress bar."""
    print(f"Downloading from: {url}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    downloaded = 0
    
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"\rProgress: {progress:.1f}% ({downloaded}/{total_size} bytes)", end='', flush=True)
    
    print("\nDownload completed!")
    return filename

def extract_zip(zip_path, extract_to):
    """Extract zip file to specified directory."""
    print(f"Extracting {zip_path} to {extract_to}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of files in the zip
        file_list = zip_ref.namelist()
        print(f"Found {len(file_list)} files in archive")
        
        # Extract all files
        zip_ref.extractall(extract_to)
    
    print("Extraction completed!")

def main():
    if len(sys.argv) != 2:
        print("Usage: python download_and_extract_xsim.py <box_download_url>")
        print("Example: python download_and_extract_xsim.py 'https://cornell.box.com/shared/static/....'")
        sys.exit(1)
    
    box_url = sys.argv[1]
    
    # Use the proper PACKAGE_ASSET_DIR from mani_skill
    assets_dir = Path(PACKAGE_ASSET_DIR)
    print(f"Assets directory: {assets_dir}")
    
    # Create assets directory if it doesn't exist
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary directory for download
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_zip = os.path.join(temp_dir, "xsim_assets.zip")
        
        try:
            # Download the file
            download_file(box_url, temp_zip)
            
            # Extract to assets directory
            # The zip contains the full path "X-Sim/simulation/ManiSkill/mani_skill/assets/xsim"
            # We want to extract it so that "xsim" ends up directly in the assets directory
            extract_zip(temp_zip, temp_dir)
            
            # Find the extracted xsim directory in the temporary location
            extracted_xsim_path = None
            for root, dirs, files in os.walk(temp_dir):
                if 'xsim' in dirs:
                    potential_path = Path(root) / 'xsim'
                    # Verify this looks like the right xsim directory by checking for expected subdirs
                    if (potential_path / 'tabletop_env').exists() or (potential_path / 'kitchen_env').exists():
                        extracted_xsim_path = potential_path
                        break
            
            if extracted_xsim_path is None:
                print("❌ Error: Could not find xsim directory in extracted files")
                print("Archive contents:")
                for root, dirs, files in os.walk(temp_dir):
                    level = root.replace(temp_dir, '').count(os.sep)
                    indent = ' ' * 2 * level
                    print(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 2 * (level + 1)
                    for file in files[:5]:  # Show only first 5 files per directory
                        print(f"{subindent}{file}")
                    if len(files) > 5:
                        print(f"{subindent}... and {len(files) - 5} more files")
                sys.exit(1)
            
            # Copy the xsim directory to the correct location
            target_xsim_path = assets_dir / "xsim"
            
            # Remove existing xsim directory if it exists
            if target_xsim_path.exists():
                print(f"Removing existing xsim directory: {target_xsim_path}")
                shutil.rmtree(target_xsim_path)
            
            # Copy the extracted xsim directory
            print(f"Copying xsim directory from {extracted_xsim_path} to {target_xsim_path}")
            shutil.copytree(extracted_xsim_path, target_xsim_path)
            
            # Verify extraction
            if target_xsim_path.exists():
                print(f"✓ Successfully extracted xsim assets to: {target_xsim_path}")
                
                # List contents to verify
                contents = list(target_xsim_path.rglob("*"))
                print(f"✓ Found {len(contents)} items in xsim directory")
                
                # Show top-level directories
                top_dirs = [item for item in target_xsim_path.iterdir() if item.is_dir()]
                if top_dirs:
                    print("Top-level directories:")
                    for dir_path in sorted(top_dirs):
                        print(f"  - {dir_path.name}")
                
                # Verify expected directories exist
                expected_dirs = ['tabletop_env', 'kitchen_env']
                missing_dirs = [d for d in expected_dirs if not (target_xsim_path / d).exists()]
                if missing_dirs:
                    print(f"⚠ Warning: Expected directories not found: {missing_dirs}")
                else:
                    print("✓ All expected directories found")
            else:
                print("❌ Error: xsim directory not found after extraction")
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error downloading file: {e}")
            sys.exit(1)
        except zipfile.BadZipFile as e:
            print(f"❌ Error: Downloaded file is not a valid zip file: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main() 