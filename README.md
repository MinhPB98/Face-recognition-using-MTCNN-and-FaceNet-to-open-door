# Face-Door
# Using face recognition to open door
1. First is git clone this github https://github.com/phanbaominh111198/face_cpu
2. Second you will create the folder called " Dataset ", subfolder called"FaceData" ( which is created by yourself too) . There are two sub-folder called "Raw" which is use to save your original image, I mean the image of yourself. And the "Processeed" to use save the image after processing such as align it and crop it into a new size
  Ex: |- Dataset
        |- FaceData
          |---processed
             |-----Minh
             |-----Minh'friend
          |---raw
             |-----Minh
             |-----Minh's friend
   In each folder which is named by yourself and the others maybe your friends, your girl friend or someone who you want to be the recognition
 
 3. cd to the file face_cpu ( I named this because I distinguish with running on gpu, but not doing it on gpu yet), and run "pip install -r requirements.txt" on the terminal. I suggest to use Tensorflow 1.15 and numpy 1.16. It will reduce the error so much 
 
 4. Then run this command "python src/align_dataset_mtcnn.py  Dataset/FaceData/raw Dataset/FaceData/processed --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25" on the terminal
    After running this step you will have a face is cropped in the folder processed
 
 5. 
