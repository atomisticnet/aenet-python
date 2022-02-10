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
 Usage: aenet-rmse.sh [options] [<train.out>]

 Grep MSE and RMSE epoch lines from 'train.x' output files
 <train.out>.  If the name of the output file is not given, a number
 of default names will be tried.

 Options:
    -h, --help        Show this help message.
"

set -o errexit
set -o pipefail

basedir=$(dirname $0)
source "${basedir}/aenet-tools.sh"

if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
  echo "$usage"
  exit 0
fi

if [[ $# > 0 ]] && [[ "$1" != "" ]]; then
  train_out="$1"
else
  train_out="$(_aenet_get_train_out)"
fi

if [[ "${train_out}" == "" ]]; then
  >&2 echo "No 'train.x' output file found and none specified."
  exit 1
fi

if ! [[ -f "${train_out}" ]]; then
  >&2 echo "File not found: ${train_out}"
  exit 2
fi

grep -B1 "<" "${train_out}"

exit 0
