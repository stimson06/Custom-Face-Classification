
from tensorflow.keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from glob import glob, iglob
import mediapipe as mp
import numpy as np
import pickle
import cv2 
import os 
import math


class Profile():
            
    def user_video_profile(self,User, BASEDIR_video):

        #creating the directory
        os.makedirs(os.path.join(BASEDIR_video,User),exist_ok=True)
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID') #VideoWriter object
        out = cv2.VideoWriter(os.path.join(BASEDIR_video,User,User+'.avi'), fourcc, 30.0, (640,  480))
        print('[INFO] Capturing the user profile...')

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("[ERROR] Frame not recieved, Please check the camera or Restart the kernel.")
                break
            out.write(frame)
            cv2.imshow('frame', frame)

            #Keyboard interupt
            if cv2.waitKey(20)== ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def user_image_profile(self,video, image_path):

        print('[INFO] Creating the images profile for {}' .format(os.path.basename(image_path)))

        #Face detection variables
        mpFaceDetection = mp.solutions.face_detection
        face_detection = mpFaceDetection.FaceDetection(min_detection_confidence=0.55)

        counter = 0        
        margin = 70
        cap = cv2.VideoCapture(video)

        while cap.isOpened():

            ret, frame = cap.read()
            results = face_detection.process(frame)
            counter +=1

            if results.detections:
                for id, detection in enumerate(results.detections):
                    bound_box_c = detection.location_data.relative_bounding_box
                    img_h, img_w, _ = frame.shape
                    (x, y, w, h) = int(bound_box_c.xmin * img_w ), int(bound_box_c.ymin * img_h ), int(bound_box_c.width * img_w ), int(bound_box_c.height * img_h )
                    
                    #Boundaries for cropping the image
                    x_a = x - margin
                    y_a = y - margin
                    x_b = x + w + margin
                    y_b = y + h + margin
                    if x_a < 0:
                        x_b = min(x_b - x_a, img_w-1)
                        x_a = 0
                    if y_a < 0:
                        y_b = min(y_b - y_a, img_h-1)
                        y_a = 0
                    if x_b > img_w:
                        x_a = max(x_a - (x_b - img_w), 0)
                        x_b = img_w
                    if y_b > img_h:
                        y_a = max(y_a - (y_b - img_h), 0)
                        y_b = img_h
                    cropped = frame[y_a: y_b, x_a: x_b]
                    resized_img = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_AREA)
                    resized_img = np.array(resized_img)
                    cv2.imwrite(image_path+'/'+os.path.basename(video)+str(counter)+'.png',resized_img)
                    

            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            cv2.imshow(os.path.basename(video), frame)

            #keyboard interupt
            if cv2.waitKey(40) == ord('q'):
                break

             #Break when the video gets completed   
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    break

        print('[INFO] Compeleted ....')
        cap.release()
        cv2.destroyAllWindows()

class Precompting():
    
    def cal_mean_feature(self, image_folder, model):
        
        print('[INFO] Generating features of {}'.format(os.path.basename(image_folder)))
        #list of images in the folder
        face_images = list(iglob(os.path.join(image_folder, '*.*')))
        
        
        def chunks(l, n):
            
            #creating list that contains the size of batch (i.e. 32)
            for i in range(0, len(l), n):
                yield l[i:i + n]

        
            #preprocessing of the images
        def image2x(image_path):

            img = image.load_img(image_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = utils.preprocess_input(x, version=1)  # or version=2
            return x

        batch_size = 32
        face_images_chunks = chunks(face_images, batch_size)
        fvecs = None
        
        for face_images_chunk in face_images_chunks:
            images = np.concatenate([image2x(face_image) for face_image in face_images_chunk])
            batch_fvecs = model.predict(images)
           
            if fvecs is None:
                fvecs = batch_fvecs
            else:
                fvecs = np.append(fvecs, batch_fvecs, axis=0)
        
        return np.array(fvecs).sum(axis=0) / len(fvecs)

    def pickle_stuff(self, filename, stuff):
        
        #Dumping the name and features into a pickle file
        save_stuff = open(filename, "wb")
        pickle.dump(stuff, save_stuff)
        save_stuff.close()
    
def main():
    
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')  # pooling: None, avg or max
    
    #paths of base directories
    BASEDIR_video = './Data/user_videos'
    BASEDIR_image = './Data/user_images'
    
    #class variables
    data = Profile()
    process = Precompting()
    
    #Customer name
    #user = input('Enter the name of customer :').lower()
    
    #video profile
    #data.user_video_profile(user, BASEDIR_video)
    
    #image profile
    folders = list(iglob(os.path.join(BASEDIR_video,'*')))
    names = [os.path.basename(folder) for folder in folders]
    for i, folder in enumerate(folders):
        name = names[i]
        videos = list(iglob(os.path.join(folder,'*')))
        save_folder = os.path.join(BASEDIR_image, name)
        os.makedirs(save_folder, exist_ok=True)
        for video in videos:
            
            data.user_image_profile(video, save_folder)
    
    print('\033[92m[INFO] Data generated successfully... \033[0m')
    
    #creating a pickle file with name and features
    precompute_features = []
    for i, folder in enumerate(folders):
        name = names[i]
        save_folder = os.path.join(BASEDIR_image, name)
        mean_features = process.cal_mean_feature(save_folder, model)
        precompute_features.append({"name": name, "features": mean_features})
        print('[INFO] Completed...')
        
    process.pickle_stuff("./features.pickle", precompute_features)
    print('\033[92m[INFO] Pickle file generation successful...  \033[0m')
    
    return precompute_features
    
if __name__ == '__main__':
    pickle_file = main()
    