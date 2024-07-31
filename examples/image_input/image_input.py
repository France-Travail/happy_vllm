import base64
import os
from typing import Union


class ImageInput:
    """
    Represents a class with specific methods to help to prepare a list of image inputs for vision models api call.

    Methods:
    __init__(list_image_path: Union[str, None]):
        Initializes a ImageInput instance with the provided attribute. Raises NotImplementedError if any attribute is None.
    
    _check_attributes():
        Checks if the required attribute List_image_path is not None.
        Raises NotImplementedError if the attribute is None.

    encode() -> dict:
        Encodes and returns a dictionary representation of the images encoded in base 64.

    extension() -> dict:
        Returns a dictionary of the extension of each image input (needed in the api call) : 

        "type": "image_url", 
           "image_url": {
            "url": f"data:image/png;base64,{base64_images[0]}"}
    """

    def __init__(self, list_image_path: Union[list, None]):
        self.list_image_path: list = list_image_path

    def _check_attributes(self):
        if not self.list_image_path:
            raise AttributeError("This attributes must be different to None")

    def encode(self):
        base64_images = {}
        # Open the image file and encode it as a base64 string
        for i in range(len(self.list_image_path)):
            with open(self.list_image_path[i], "rb") as image_file:
                base64_images[i] = base64.b64encode(image_file.read()).decode("utf-8")
        return base64_images

    def extension(self):
        extension_images = {}
        # Open the image file and encode it as a base64 string
        for i in range(len(self.list_image_path)):
            extension_images[i] = os.path.splitext(self.list_image_path[i])[1][1:]
        return extension_images