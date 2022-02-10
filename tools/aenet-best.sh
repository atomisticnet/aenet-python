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
 Usage: aenet-best.sh [options]

 Determine the epochs with the lowest test set RMSE.  Per default, the
 only the one best epoch will be printed out.

 Options:
    -h, --help        Show this help message.
    -n, --num-epochs  Number of epochs to be printed out
    -t, --train-out   Name of the 'train.x' output file
"

set -o errexit
set -o pipefail

# defaults
num_epochs=1
train_out=""

while [[ "$1" != "" ]]; do
    case "$1" in
	-h | --help)
	    echo "${usage}"
	    exit 0
	    ;;
	-n | --num-epochs)
	    shift
	    num_epochs=$1
            ;;
        -t | --train-out)
	    shift
	    train_out="$1"
	    ;;
	*)
	    >&2 echo "Unknown argument: $1"
	    exit 1
	    ;;
    esac
    shift
done

aenet-rmse.sh "${train_out}" | sort -g -k 5 | head -n $((${num_epochs} + 2)) | tail -n ${num_epochs}

exit 0
