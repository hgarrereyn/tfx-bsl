workspace(name = "tfx_bsl")

# To update TensorFlow to a new revision.
# 1. Update the '_TENSORFLOW_GIT_COMMIT' var below to include the new git hash.
# 2. Get the sha256 hash of the archive with a command such as...
#    curl -L https://github.com/tensorflow/tensorflow/archive/<git hash>.tar.gz | sha256sum
#    and update the 'sha256' arg with the result.
# 3. Request the new archive to be mirrored on mirror.bazel.build for more
#    reliable downloads.

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

# TF 1.14
_TENSORFLOW_GIT_COMMIT = "87989f69597d6b2d60de8f112e1e3cea23be7298"

http_archive(
    name = "org_tensorflow",
    sha256 = "c4da79385dfbfb30c1aaf73fae236bc6e208c3171851dfbe0e1facf7ca127a6a",
    strip_prefix = "tensorflow-%s" % _TENSORFLOW_GIT_COMMIT,
    urls = [
        "https://mirror.bazel.build/github.com/tensorflow/tensorflow/archive/%s.tar.gz" % _TENSORFLOW_GIT_COMMIT,
        "https://github.com/tensorflow/tensorflow/archive/%s.tar.gz" % _TENSORFLOW_GIT_COMMIT,
    ],
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

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

tf_workspace(
    path_prefix = "",
    tf_repo_name = "org_tensorflow",
)

# LINT.IfChange(arrow_version)
ARROW_COMMIT = "ff7ee06020949daf66ac05090753e1a17736d9fa"  # 0.17.1
# LINT.ThenChange(third_party/arrow/util/config.h)

http_archive(
    name = "arrow",
    build_file = "//third_party:arrow.BUILD",
    strip_prefix = "arrow-%s" % ARROW_COMMIT,
    sha256 = "2fca47067e97417d6ba0574b6b90a66752176ac4315a93e6e42d7af8c312e1c1",
    urls = ["https://github.com/apache/arrow/archive/%s.zip" % ARROW_COMMIT],
    patches = ["//third_party:arrow.patch"],
)

# Specify the minimum required bazel version.
load("@org_tensorflow//tensorflow:version_check.bzl", "check_bazel_version_at_least")

git_repository(
    name = "com_github_tensorflow_metadata",
    commit = "992edd17e0f020458084c031c42f85d520e6f6af",
    remote = "https://github.com/tensorflow/metadata.git",
)

check_bazel_version_at_least("0.24.1")
