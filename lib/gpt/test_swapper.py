import pytest
from swapper_api import image_swapper

# Testing valid inputs
def test_valid_inputs():
    assert image_swapper(source_img='./data/man1.jpeg;./data/man2.jpeg', target_img='./data/mans1.jpeg', face_restore=True, background_enhance=True, face_upsample=True, upscale=2, codeformer_fidelity=0.5) == 0

# Testing invalid source_img
def test_invalid_source_img():
    with pytest.raises(ValueError):
        image_swapper(source_img='./data/non_existent.jpeg', target_img='./data/mans1.jpeg')

# Testing invalid target_img
def test_invalid_target_img():
    with pytest.raises(ValueError):
        image_swapper(source_img='./data/man1.jpeg', target_img='./data/non_existent.jpeg')

# Testing invalid upscale value
def test_invalid_upscale_value():
    with pytest.raises(ValueError):
        image_swapper(source_img='./data/man1.jpeg;./data/man2.jpeg', target_img='./data/mans1.jpeg', upscale='invalid')

# Testing invalid codeformer_fidelity value
def test_invalid_codeformer_fidelity_value():
    with pytest.raises(ValueError):
        image_swapper(source_img='./data/man1.jpeg;./data/man2.jpeg', target_img='./data/mans1.jpeg', codeformer_fidelity='invalid')

# Testing invalid boolean inputs
def test_invalid_boolean_inputs():
    with pytest.raises(ValueError):
        image_swapper(source_img='./data/man1.jpeg;./data/man2.jpeg', target_img='./data/mans1.jpeg', face_restore='invalid', background_enhance=True, face_upsample=True)
