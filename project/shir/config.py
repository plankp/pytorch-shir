"""
All sorts of configuration flags
"""

""" Settings that you would play with """

# if we should trigger synthesis
PERFORM_SYNTHESIS = False

# the UUID of the accelerator as a bytestring
# only relevant if PERFORM_SYNTHESIS = True
ACCEL_UUID = None

# if we should simulate the hardware
PERFORM_SIMULATION = False

# the number of bits in each cacheline (of the target / simulated hardware)
CACHELINE_BITS = 512

# the output directory:
# each model will get its own subdir here
EMIT_OUTPUT_DIR = "./data/generated"

""" Settings that you normally wouldn't touch """

# the template directory
TEMPLATE_DIR = "./template"

# the shared library for driver
DRIVER_LIB = "./driver/build/libdriver.so"
