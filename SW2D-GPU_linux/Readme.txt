************************************************************************************************
 Two-dimensional shallow water model accelerated by GPGPU (SW2D-GPU)

 Sequential code developer(Original version) : Seungsoo Lee   | Code written in FORTRAN 90

 Developer of parallel code in GPGPU: Tomas Carlotto          | Code written in CUDA C/C++

************************************************************************************************
Prerequisites for using parallel code:
         Computer equipped with NVIDIA GPU (compatible with CUDA technology).
        Software required: CUDA™ Toolkit 8.0 or later 
                  
         System: Linux
************************************************************************************************
This is the version of SW2D-GPU for Linux.
The source code of the model is in the file "SW2D-GPU.cu" within the subfolder "src" 
To run SW2D-GPU.cu with any version of the CUDA toolkit, it is recommended 
to simply enter the "src" subfolder and use the "make" command to compile 
the code and then use ./SW2D-GPU to run the examples provided in the folder "db".