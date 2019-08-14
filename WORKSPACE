workspace(name = "tfx_common")

# To update TensorFlow to a new revision.
# 1. Update the '_TENSORFLOW_GIT_COMMIT' var below to include the new git hash.
# 2. Get the sha256 hash of the archive with a command such as...
#    curl -L https://github.com/tensorflow/tensorflow/archive/<git hash>.tar.gz | sha256sum
#    and update the 'sha256' arg with the result.
# 3. Request the new archive to be mirrored on mirror.bazel.build for more
#    reliable downloads.

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# TF 1.14
_TENSORFLOW_GIT_COMMIT = "87989f69597d6b2d60de8f112e1e3cea23be7298"
http_archive(
    name = "org_tensorflow",
    sha256 = "c4da79385dfbfb30c1aaf73fae236bc6e208c3171851dfbe0e1facf7ca127a6a",
    urls = [
      "https://mirror.bazel.build/github.com/tensorflow/tensorflow/archive/%s.tar.gz" % _TENSORFLOW_GIT_COMMIT,
      "https://github.com/tensorflow/tensorflow/archive/%s.tar.gz" % _TENSORFLOW_GIT_COMMIT,
    ],
    strip_prefix = "tensorflow-%s" % _TENSORFLOW_GIT_COMMIT,
)


# TensorFlow depends on "io_bazel_rules_closure" so we need this here.
# Needs to be kept in sync with the same target in TensorFlow's WORKSPACE file.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "e0a111000aeed2051f29fcc7a3f83be3ad8c6c93c186e64beb1ad313f0c7f9f9",
    strip_prefix = "rules_closure-cf1e44edb908e9616030cc83d085989b8e6cd6df",
    urls = [
        "http://mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/cf1e44edb908e9616030cc83d085989b8e6cd6df.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/cf1e44edb908e9616030cc83d085989b8e6cd6df.tar.gz",  # 2019-04-04
    ],
)

# Needed by pybind_extension rule from Tensorflow.
# When upgrading tensorflow version, also check tensorflow/WORKSPACE for the
# version of this -- keep in sync.
http_archive(
    name = "bazel_skylib",
    sha256 = "2ef429f5d7ce7111263289644d233707dba35e39696377ebab8b0bc701f7818e",
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/0.8.0/bazel-skylib.0.8.0.tar.gz"],
)

http_archive(
    name = "tf_custom_op",
    strip_prefix = "custom-op-b74be358b804ee563a011cb255642ec17c6bbc1a",
    sha256 = "048e74e977ab3c2bd3f43e460f9c4285c4fecf4154f75953ca4db0807ec593cf",
    urls = ["https://github.com/tensorflow/custom-op/archive/b74be358b804ee563a011cb255642ec17c6bbc1a.zip"],
)

load("@tf_custom_op//tf:tf_configure.bzl", "tf_configure")
tf_configure(name = "local_config_tf")

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
tf_workspace(
    path_prefix = "",
    tf_repo_name = "org_tensorflow",
)

load("//third_party:arrow_configure.bzl", "arrow_configure")
arrow_configure(name = "arrow")

# Specify the minimum required bazel version.
load("@org_tensorflow//tensorflow:version_check.bzl", "check_bazel_version_at_least")

check_bazel_version_at_least("0.24.1")
