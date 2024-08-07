import pytest

from happy_vllm.utils import ImageInput


def test_imageinput():
    # Test ImageInput class creation
    image_input = ImageInput(image_paths=["./test.jpg"])
    assert image_input.image_paths == ["./test.jpg"]

    # Test ImageInput class encode method
    file_base64 = open('./test_image_base64.txt')
    test_image_base64 = file_base64.read()
    file_base64.close()
    expected_base64 = {0: test_image_base64}
    assert image_input.base64_images == expected_base64

    # Test ImageInput class extension method
    expected_extension = {0: 'jpg'}
    assert image_input.extension_images == expected_extension