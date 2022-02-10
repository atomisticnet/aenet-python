#!/usr/bin/env bash
# This file is part of the AENET package.
#
# Copyright (C) 2019 Nongnuch Artrith
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
usage="
 Usage: aenet-get-best-ann.sh [options] <directory>

 Copy the ANN potentials that correspond to the lowest test set RMSE
 from directory <directory> to the current directory.

 Options:
    -h, --help        Show this help message.
"

# set -o errexit
set -o pipefail

basedir=$(dirname $0)
source "${basedir}/aenet-tools.sh"

if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
  echo "$usage"
  exit 0
fi

if [[ $# > 0 ]]; then
  srcdir="$1"
else
  >&2 echo "No source directory specified"
  exit 1
fi

if ! [[ -d "${srcdir}" ]]; then
  >&2 echo "Source directory not found: ${srcdir}"
  exit 2
fi

curdir="$(pwd)"
cd "${srcdir}"
best_epoch=$(aenet-best.sh | awk '{printf("%05d", $1)}')
ann_pots="$(_aenet_get_ann_pot_names)"
cd "${curdir}"
for pot in ${ann_pots}; do
  ( set -x; cp -p -i "${srcdir}/${pot}-${best_epoch}" . )
  ( set -x; cp -i "./${pot}-${best_epoch}" "./${pot}" )
done

exit 0
