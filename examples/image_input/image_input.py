import base64
import os
from typing import Union


class ImageInput:
    """
    Represents a class with specific methods to help to prepare a list of image inputs for vision models api call.

    Methods:
    __init__(image_paths: list):
        Initializes a ImageInput instance with the provided attribute. Raises NotImplementedError if any attribute is None.

    encode() -> dict:
        Encodes and returns a dictionary representation of the images encoded in base 64.

    extension() -> dict:
        Returns a dictionary of the extension of each image input (needed in the api call) : 

        "type": "image_url", 
           "image_url": {
            "url": f"data:image/png;base64,{base64_images[0]}"}
    """

    def __init__(self, image_paths: list):
        self.image_paths: list = image_paths.copy()
        self.base64_images = self.__encode()
        self.extension_images = self.__extension()

    def _encode(self) -> dict:
        """Opens the image files and encode it as a base64 string
        
        Returns:
            dict : The base 64 of all images
        """

        base64_images = {}
        for i in range(len(self.image_paths)):
            with open(self.image_paths[i], "rb") as image_file:
                base64_images[i] = base64.b64encode(image_file.read()).decode("utf-8")
        return base64_images

    def _extension(self) -> dict:
        """Extracts the string of the extension of the image files
        
        Returns:
            dict : The extension of all images
        """

        extension_images = {}
        for i in range(len(self.image_paths)):
            extension_images[i] = os.path.splitext(self.image_paths[i])[1][1:]
        return extension_images