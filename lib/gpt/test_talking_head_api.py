import pytest
from talking_head_api import talking_head

def test_talking_head():
    # Testing for boolean input in enhancer
    with pytest.raises(ValueError):
        talking_head('audio.wav', 'image.png', 'results', enhancer='True')
    # Testing for boolean input in still
    with pytest.raises(ValueError):
        talking_head('audio.wav', 'image.png', 'results', still='False')
    # Testing for correct preprocess input
    with pytest.raises(ValueError):
        talking_head('audio.wav', 'image.png', 'results', preprocess='other')
    # Testing for positive float in expression_scale
    with pytest.raises(ValueError):
        talking_head('audio.wav', 'image.png', 'results', expression_scale=-0.5)
    # Testing for successful run
    assert talking_head('audio.wav', 'image.png', 'results') is None
