set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# if nvjitlink not in LD_LIBRARY_PATH, add it
if [[ ":${LD_LIBRARY_PATH:-}:" != *":$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/nvjitlink/lib')"):"* ]]; then
    export LD_LIBRARY_PATH="$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/nvjitlink/lib')"):${LD_LIBRARY_PATH:-}"
fi

cmake --fresh \
    -S "$SCRIPT_DIR" \
    -B "$SCRIPT_DIR/build" \
    -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
    -DUSE_SYSTEM_NVTX:BOOL=ON
cmake --build "$SCRIPT_DIR/build" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"
