#!/usr/bin/env bash
set -euo pipefail
REV_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REV_DEPS="${REV_ROOT}/tmp/revision_dependencies"
# Dependency download/build only. No changes to system or frozen solver sources.
mkdir -p "${REV_DEPS}/packages" "${REV_DEPS}/prefix"
if [[ ! -d "${REV_DEPS}/trac_ik/.git" ]]; then
  git clone https://github.com/traclabs/trac_ik.git "${REV_DEPS}/trac_ik"
fi
git -C "${REV_DEPS}/trac_ik" checkout --detach 90162ac2ecc6ea8f88c6e99df6ee01efd217a3fb
cd "${REV_DEPS}/packages"
apt-get download libnlopt-cxx-dev=2.7.1-3build1 libnlopt-dev=2.7.1-3build1 libnlopt0=2.7.1-3build1 libnlopt-cxx0=2.7.1-3build1
for REV_DEB in ./*.deb; do dpkg-deb -x "${REV_DEB}" "${REV_DEPS}/prefix"; done
cmake -S "${REV_ROOT}/src/confik/revision_compute_allocation/native" -B "${REV_DEPS}/build" \
  -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DTRAC_SOURCE="${REV_DEPS}/trac_ik" -DCMAKE_PREFIX_PATH="${REV_DEPS}/prefix/usr;/opt/ros/humble"
cmake --build "${REV_DEPS}/build" -j 4
