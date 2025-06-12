#!/bin/bash

# X-Sim Environment Setup Script
# This script automates the complete environment setup for X-Sim


echo "🚀 Starting X-Sim Environment Setup..."
echo "======================================"

echo "📦 Step 1: Creating conda environment..."
echo "----------------------------------------"

echo "Creating new conda environment 'xsim' with Python 3.10..."
conda create -n xsim python=3.10

echo "✅ Conda environment 'xsim' created successfully!"

echo ""
echo "📋 Step 2: Installing ManiSkill simulation environment..."
echo "--------------------------------------------------------"

echo "Activating conda environment and installing ManiSkill..."
conda activate xsim

# Install ManiSkill in editable mode
pip install -e simulation/ManiSkill/

echo "✅ ManiSkill installation completed!"

echo ""
echo "💾 Step 3: Downloading X-Sim dataset..."
echo "---------------------------------------"

echo "Downloading X-Sim assets and dataset..."
python simulation/ManiSkill/download_and_extract_xsim.py "https://cornell.box.com/shared/static/qjfltim1ca96co8zdkhswes9cdnpbiap.zip"

echo "✅ Dataset download completed!"

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "Your X-Sim environment is now ready to use!"
echo ""
echo "To activate the environment:"
echo "  conda activate xsim"
echo ""
echo "To run the full pipeline (Example):"
echo "  python run_pipeline.py --env_id \"Mustard-Place\""
echo ""
echo "To run individual steps, see the README.md for detailed usage."
echo ""