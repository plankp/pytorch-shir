"""
All sorts of configuration flags
"""

# if we should trigger synthesis
PERFORM_SYNTHESIS = False

# the UUID of the accelerator as a bytestring
# only relevant if PERFORM_SYNTHESIS = True
ACCEL_UUID = None

# if we should simulate the hardware
PERFORM_SIMULATION = False

# the number of bits in each cacheline (of the target / simulated hardware)
CACHELINE_BITS = 512

# if we should force the generation of hardware files even if it is not needed
FORCE_GENERATE_FILES = True

# the output directory:
# each model will get its own subdir here
EMIT_OUTPUT_DIR = "./data/generated"

# the template directory
# YOU shouldn't be touching this!
TEMPLATE_DIR = "./template"
