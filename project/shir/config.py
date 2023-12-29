"""
All sorts of configuration flags
"""

""" Settings that you would play with """

# if we should trigger synthesis
PERFORM_SYNTHESIS = False

# the UUID of the accelerator as a bytestring
# only relevant if PERFORM_SYNTHESIS
ACCEL_UUID = None

# the number of bits in each cacheline (of the target / simulated hardware)
CACHELINE_BITS = 512

# if FPGA execution should print runtime statistics
FPGA_PRINT_RTINFO = True

# if we should try to reduce bit width of known tensor values
TRY_NARROW_TYPE = True

# if we should copy the known tensor values ahead of time
TRY_COPY_AOT = True

# the model cache directory:
MODEL_CACHE_DIR = "./data/cache"

""" Settings that you normally wouldn't touch """

# the template directory
TEMPLATE_DIR = "./template"

# the shared library for driver
# only relevant if PERFORM_SYNTHESIS
DRIVER_LIB = "./driver/build/libdriver.so"
