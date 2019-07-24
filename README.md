# ODcode
Few scripts to use google object detection API

# Input
Read .json files per image of the form:
```
{
    "imageHeight": 512,
    "imageWidth": 512,
    "bboxes": [
        {
            "id": "42a2b776-f106-48e5-80c8-0cf365da2387",
            "left": 41,
            "top": 82,
            "width": 12,
            "height": 12,
            "confidence": 100,
            "label": "rock"
        },
        {
            "id": "f09d87ab-93c0-4e30-9b22-205f5a51ee12",
            "left": 127,
            "top": 1,
            "width": 17,
            "height": 19,
            "confidence": 100,
            "label": "rock"
        }
    ]
}
```
