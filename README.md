# Face-Door
# Using face recognition to open door
# 1. First is git clone this github https://github.com/phanbaominh111198/face_cpu
# 2. Second you will create the folder called " Dataset ", subfolder called"FaceData" ( which is created by yourself too) . There are two sub-folder called "Raw" which is use to save your original image, I mean the image of yourself. And the "Processeed" to use save the image after processing such as align it and crop it into a new size
  Ex: |- Dataset
        |- FaceData
          |---processed
             |-----Minh
             |-----Minh'friend
          |---raw
             |-----Minh
             |-----Minh's friend
   In each folder which is named by yourself and the others maybe your friends, your girl friend or someone who you want to be the recognition
 
 # 3. cd to the file face_cpu ( I named this because I distinguish with running on gpu, but not doing it on gpu yet), and run command below on the terminal. I suggest to use Tensorflow 1.15 and numpy 1.16. It will reduce the error so much 
      pip install -r requirements.txt
 
 # 4. Then run this command on the terminal
    python src/align_dataset_mtcnn.py  Dataset/FaceData/raw Dataset/FaceData/processed --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25
 After running this step you will have a face is cropped in the folder processed
 
 # 5. Create the folder named "models" and download this pretrain to the folder. Link to download pretrain https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view .
 Remembering to delete the "facemodel.pkl" file
 # 6. Run this command 
    python src/classifier.py TRAIN Dataset/FaceData/processed Models/20180402-114759.pb Models/facemodel.pkl --batch_size 1000
 when this appear the line “Saved classifier model to file “Models/facemodel.pkl”. It is done 
 # 7.python src/face_rec.py 
 Run and test the recognition 
 # Additional for someone who one to continue to control relay or door. Just running the code in the folder "serial-communication" on arduino and then running the code face_rec.py
   If someone who just one to run the face recognition only. Just run the code face_rec.py and put all the comment relating to the serial into the comment.
