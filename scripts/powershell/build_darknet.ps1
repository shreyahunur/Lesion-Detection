# Powershell script uses vcpkg to install darknet: opencv-base, cuda and cudnn
# for x64-windows; Then goes to darknet/ dir and runs its ps1 script to build it

# I am running this script inside Notebooks/ dir from YOLOv4 Notebook
# so we dont need to include Notebooks/ here
# cd yolov4\vcpkg
# $env:VCPKG_ROOT=$PWD
# .\bootstrap-vcpkg.bat
# # for a quicker install of dependencies
# .\vcpkg install darknet[opencv-base,cuda,cudnn]:x64-windows
# cd ..
Set-ExecutionPolicy unrestricted -Scope CurrentUser -Force
cd yolov4\darknet
.\build.ps1 -UseVCPKG -EnableOPENCV -EnableCUDA -EnableCUDNN -DisableInteractive
