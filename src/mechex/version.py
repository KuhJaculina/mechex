"""Release identity."""

__version__ = "1.0.0"
__release__ = "mechex-1.0.0"

MODEL_NAME = "MechEx"

# The release directory name. The manuscript title exceeds the code host's repository-name
# limit, so the release is published and checked out under a short slug; recording the slug
# rather than the local directory keeps the verification artefacts byte-identical in the
# working tree, in a clone of the published repository and in the container image.
RELEASE_DIR = "mechex"
