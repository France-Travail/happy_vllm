# Image Input

## Add image in put in a prompt
To avoid repeatedly encoding an image and extracting the related extension, you can create a class inheriting from ```image_input.ImageInput```. 

You need to instantiate the following attributes  : path of the image file (string)

## How to use
After deploying your REST API, you just need to first call it with the method encode() and extension(), and then with the following route ```/v1/chat/completions``` and a body as follows:

```
{"messages": [
      {
        "role": "user",
        "content": [
            {"type": "text", "text": "Summarize all the information included in this image."},
            {  "type": "image_url",
                "image_url": {
                    "url": "data:image/{image_extension[0]};base64,{image_base64[0]}"
                }
            }
         ]
      }
    ],
    "model": "llava-v1.6-mistral-7b-hf",
    "max_tokens": 1024,
    "temperature":0.0
  }'''
```

## Warning

From vllm 0.5.0 to 0.5.3.post1, only one image can be passed per call.

