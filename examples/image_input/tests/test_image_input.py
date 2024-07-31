import pytest

from happy_vllm.utils import ImageInput


def test_imageinput():
    # Test ImageInput class creation
    image_input = ImageInput(list_image_path=["./test.jpg"])
    assert image_input.list_image_path == ["./test.jpg"]

    # Test ImageInput class encode method
    file_base64 = open('./test_image_base64.txt')
    test_image_base64 = file_base64.read()
    file_base64.close()
    expected_base64 = {0: test_image_base64}
    assert image_input.encode() == expected_base64

    # Test ImageInput class extension method
    expected_extension = {0: 'jpg'}
    assert image_input.extension() == expected_extension

    # Test ImageInput class creation with missing parameters
    with pytest.raises(AttributeError, match="This attribute must be different to None"):
        ImageInput(list_image_path=None)