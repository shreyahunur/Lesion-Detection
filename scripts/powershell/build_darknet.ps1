# Powershell script uses vcpkg to install darknet: opencv-base, cuda and cudnn
# for x64-windows; Then goes to darknet/ dir and runs its ps1 script to build it
cd Notebooks\yolov4\vcpkg
$env:VCPKG_ROOT=$PWD
.\bootstrap-vcpkg.bat
# for a quicker install of dependencies
.\vcpkg install darknet[opencv-base,cuda,cudnn]:x64-windows
cd ..
cd darknet
.\build.ps1
