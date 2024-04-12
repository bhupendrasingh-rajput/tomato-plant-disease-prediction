#Import necessary libraries
from flask import Flask, render_template, request
 
import numpy as np
import os
 
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
 
#load model
model =load_model("model/model100epoch.h5")
 
print('@@ Model loaded')
 
 
def pred_cot_dieas(cott_plant):
  test_image = load_img(cott_plant, target_size = (150, 150)) # load image 
  print("@@ Got Image for prediction")
   
  test_image = img_to_array(test_image)/255 # convert image to np array and normalize
  test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
   
  result = model.predict(test_image).round(3) # predict diseased palnt or not
  print('@@ Raw result = ', result)
   
  pred = np.argmax(result) # get the index of max value
 
  if pred == 0:
    return "Bacterial Spot", '1.html' # if index 0 burned leaf
  elif pred == 1:
      return 'Early Blight', '2.html' # # if index 1
  elif pred == 2:
      return 'Late Blight', '3.html'  # if index 2  fresh leaf
  elif pred ==3 :
    return "Leaf Mold" , '4.html' # if index 3
  elif pred ==4 :
    return "Septoria Leaf Spot", '5.html'
  elif pred ==5 :
    return "Two Spotted Spider-mites", '6.html'
  elif pred ==6 :
    return "Target Spot", '7.html'
  elif pred ==7 :
    return "Yellow Leaf Curl Virus", '8.html'
  elif pred ==8 :
    return "Tomato Mosaic Virus", '9.html'
  else :
    return "Healthy", '10.html'

 
#------------>>pred_cot_dieas<<--end
     
 
# Create flask instance
app = Flask(__name__)
 
# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('index.html')
     
  
# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']  # get input
        filename = file.filename
        print("@@ Input posted = ", filename)

        # Create the 'static/user uploaded' directory if it doesn't exist
        upload_directory = 'static/user uploaded'
        os.makedirs(upload_directory, exist_ok=True)

        file_path = os.path.join(upload_directory, filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred, output_page = pred_cot_dieas(cott_plant=file_path)

        return render_template(output_page, pred_output=pred, user_image=file_path)


     
# For local system &amp; cloud
if __name__ == "__main__":
    app.run(threaded=False)