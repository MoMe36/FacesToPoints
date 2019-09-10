# Faces to Points 


This is a fork from [cleardusk repo](https://github.com/cleardusk/3DDFA.git) which I modified so that the detector returns normalized facial landmark 3D coordinates in numpy matrix format of shape 68x3. 

### Usage

First, download dlib landmark pre-trained model on [Google Drive](https://drive.google.com/open?id=1kxgOZSds1HuUIlvo5sRH3PJv377qZAkE), create a `models` folder and put it inside. Then, run: 


```bash
python faces_to_points --f=/path/to/images
```

This will create a `results` folder and put the matrices inside. 


