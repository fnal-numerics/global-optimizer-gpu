add_executable(demo demo.cu)
set_property(TARGET demo PROPERTY CUDA_ARCHITECTURES 70-real)
set_target_properties(demo PROPERTIES POSITION_INDEPENDENT_CODE ON)
