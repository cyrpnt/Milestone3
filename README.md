# Deblurring images

## How to build the environment needed? 

Run the following commands: 
- `docker build --tag dnn ./`
- `docker run --network host dnn:latest python3 api.py`

Be careful, you need some space to build all of it and have the dataset on your computer. You will need:
- **918.5 MB** for the code and dataset
- **883MB** for the container's image (python 3.8.10)
- **4.12GB** for the container you will build

For a total of approximately **6 GB**.


## How to use it? 

Then open a navigator and go the URL `http://127.0.0.1:5000` or `http://localhost:5000` (same thing)

You can then select a picture and upload it to the website. When you deblur it, an image will be saved to your computer.

Images are going to be converted to RGB format.
