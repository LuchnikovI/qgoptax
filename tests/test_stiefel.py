import pytest
from tests.test_manifolds import CheckManifolds
from jax import random
from qgoptax.manifolds.stiefel_manifold import StiefelManifold
from jax.config import config
config.update("jax_enable_x64", True)

@pytest.fixture(params=['StiefelManifold'])
def stiefel_name(request):
    return request.param

@pytest.fixture(params=[(8, 4)])
def stiefel_shape(request):
    return request.param

@pytest.fixture(params=[1.e-6])
def stiefel_tol(request):
    return request.param

@pytest.fixture(params=['euclidean', 'canonical'])
def stiefel_metric(request):
    return request.param

@pytest.fixture(params=['svd', 'cayley', 'qr'])
def stiefel_retraction(request):
    return request.param

def test_stiefel_manifold(stiefel_name, stiefel_metric, stiefel_retraction, stiefel_shape, stiefel_tol):
    Test = CheckManifolds(
        random.PRNGKey(42),
        StiefelManifold(metric=stiefel_metric, retraction=stiefel_retraction),
        (stiefel_name, stiefel_metric),
        stiefel_shape,
        stiefel_tol
    )
    Test.checks()
