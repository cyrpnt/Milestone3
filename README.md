# Deblurring images

## How to build the environment needed? 

Run the following commands: 
- `docker build --tag dnn .`
- `docker run -p 5000:5000 dnn:latest`

Be careful, you need some space to build all of it and have the dataset on your computer. You will need:
- **21 kB** for the code
- **883MB** for the container's image (python 3.8.10)
- **4.11GB** for the container you will build

For a total of approximately **5 GB**.

If you don't have enough space because you have build too many images you can do:
- `docker image list` to see the images that are on your computer 
- `docker image rm -f <IMAGE_ID>` with the ID of the image you do not use anymore. If you remove the wrong image you will have to rebuild it
- `docker container prune -f` to remove all cached data (which are **not** removed when you just do a rm command)

If you do these commands you will get the space that was lost before.

## How to use it? 

Then open a navigator and go the URL `http://127.0.0.1:5000` or `http://localhost:5000` (same thing)

You can then select a picture and upload it to the website. When you deblur it, an image will be saved to your computer.

Images are going to be converted to RGB format.
