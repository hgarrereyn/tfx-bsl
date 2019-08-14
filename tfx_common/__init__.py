# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Init module for tfx_common."""

# tfx_common's extension module depends on pyarrow's and tensorflow's shared
# libraries. Importing the corresponding python packages will cause those
# libraries to be loaded. This way the dynamic linker wouldn't need to search
# for those libraries in the filesystem (which is bound to fail).
from tfx_common import pyarrow_tf as _
