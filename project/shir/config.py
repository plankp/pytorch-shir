"""
All sorts of configuration flags
"""

# if we should simulate the hardware
PERFORM_SIMULATION = False

# the number of bits in each cacheline (of the target / simulated hardware)
CACHELINE_BITS = 512

# whether we want to emit (and compile) SHIR code
EMIT_SHIR_CODE = True

# whether we want to emit the input tensors as CSV
EMIT_DATA_FILES = True

# the output directory:
# each model will get its own subdir here
EMIT_OUTPUT_DIR = "./data/generated"

# the template directory
# YOU shouldn't be touching this!
TEMPLATE_DIR = "./template"
