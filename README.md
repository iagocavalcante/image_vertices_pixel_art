# Generate vertices for images

This projects emerge from a personal project built with flutter and a needed to create a polygon hitbox to improve collision
for the players, the output is an image with highlighted version of the input image and a Vector2 array for flutter
flame, then its just need to copy the array and put into the flame array list for vertices.

# How to run

Download the project, with the python3 installed, all we need to do is install deps with:

> `pip install -r requirements.txt`

After installation is done, run the follow command:

The first argument shoudl be a image path or image url:

> `python cli.py ./input1.png --output result.png`
