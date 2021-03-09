************************************************************************************************
 Two-dimensional shallow water model accelerated by GPGPU (SW2D-GPU)

 Sequential code developer(Original version) : Seungsoo Lee   | Code written in FORTRAN 90

 Developer of parallel code in GPGPU: Tomas Carlotto          | Code written in CUDA C/C++

************************************************************************************************
Prerequisites for using parallel code:
         Computer equipped with NVIDIA GPU (compatible with CUDA technology).
        Software required: CUDA™ Toolkit 8.0 or later 
                  
         System: Windows
         To view and edit the code you must have:
                  Visual Studio community 2013 or later version
************************************************************************************************

This is the version of SW2D-GPU for Visual Studio community 2013 or later version (windows system).
The source code of the model is in the file "SW2D-GPU.cu" within the subfolder "SW2D-GPU". 
To run SW2D-GPU.cu with any version of the Visual Studio community it is recommended to create a new project 
in Visual Studio (2013 or later) and replace the code "kernel.cu" with the code that is in "SW2D-GPU.cu". 

