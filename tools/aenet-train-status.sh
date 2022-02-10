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
 Usage: aenet-train-status.sh <dir1> [<dir1> <dir3> ...]

 Print best-so-far epoch of the training runs in a list of
 directories.

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

curdir=$(pwd)
for d in "$@"; do
  cd "$d"
  train_out="$(_aenet_get_train_out)"
  if [[ -s "${train_out}" ]]; then
    best="$(aenet-best.sh)"
    iter_done="$(aenet-rmse.sh | awk 'END{print $1}')"
    iter_total="$(awk '/Number of iterations    :/{print $(NF)}' "${train_out}")"
    if [[ $(grep -c "STOP" "${train_out}") > 0 ]]; then
        status="(Stopped after ${iter_done} of ${iter_total} iterations)"
    elif [[ $(grep -c "training done." "${train_out}") > 0 ]]; then
        status="(Completed after ${iter_done} iterations)"
    else
        status="(Running: ${iter_done} of ${iter_total} iterations done)"
    fi
    echo "${d} ${best} ${status}"
  fi
  cd "${curdir}"
done

exit 0
