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
#-----------------------------------------------------------------------
# This file contains functions used in other scripts.

function _aenet_get_train_out () {
  train_out="$(ls -1 train.out fit-01.dat 2>/dev/null | head -n 1 || true)"
  if [[ -s "${train_out}" ]]; then
      echo "${train_out}"
  else
      echo ""
  fi
}

function _aenet_get_ann_pot_names () {
  train_out=$(_aenet_get_train_out)
  awk '/Saving the .* network to file :/{print $(NF)}' "${train_out}" | sort -u
}

