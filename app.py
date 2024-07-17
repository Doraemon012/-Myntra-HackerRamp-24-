import time
import os
import cv2
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify, request, redirect,url_for
import math
import numpy as np
import base64


products = [  {
        "id": 1,
        "name": "T-Shirt",
        "price": 500,
        "image": "product-images/tshirt.jpeg",
        "description": "A cool t-shirt",
        "model": "dress0001.glb"
      },
      {
        "id": 2,
        "name": "Dress",
        "price": 1000,
        "image": "product-images/jeans.jpeg",
        "description": "Stylish Dress",
        "model": "dress.glb",
        "sizes": [26, 28, 30, 32, 34, 36]
      },
      {
        "id": 3,
        "name": "Pink T-Shirt",
        "price": 500,
        "image": "https://www.snitch.co.in/cdn/shop/products/Snitch_April22_0161-1.jpg?v=1651234220",
        "description": "Casual Cotton T-Shirt",
        "sizes": ["S", "M", "L", "XL", "XXL"]
      },
      {
        "id": 4,
        "name": "Sneakers",
        "price": 2000,
        "image": "https://rukminim2.flixcart.com/image/850/1000/xif0q/shoe/t/3/n/3-875-mlt-3-deals4you-multi-original-imagg27xn9hnxw6h.jpeg?q=90&crop=false",
        "description": "Comfortable Running Sneakers",
        "sizes": [7, 8, 9, 10, 11, 12]
      },
      {
        "id": 5,
        "name": "Jacket",
        "price": 3000,
        "image": "https://images.bestsellerclothing.in/data/only/4-nov-2023/295499301_g0.jpg?width=1080&height=1355&mode=fill&fill=blur&format=auto",
        "description": "Warm Winter Jacket",
        "sizes": ["S", "M", "L", "XL"]
      },
      {
        "id": 6,
        "name": "Skirt",
        "price": 1200,
        "image": "https://www.tjori.com/cdn/shop/products/278420-2019-07-09-13_03_02.665543-big-image.jpg?v=1663261677",
        "description": "Elegant Midi Skirt",
        "sizes": [26, 28, 30, 32]
      },
      {
        "id": 7,
        "name": "Blouse",
        "price": 800,
        "image": "https://5.imimg.com/data5/SELLER/Default/2023/5/307478130/BW/YM/UN/134292146/multicolor-kalamkari-print-blouse.jpg",
        "description": "Chic Silk Blouse",
        "sizes": ["S", "M", "L", "XL"]
      },
      {
        "id": 10,
        "name": "Sneakers",
        "price": 2000,
        "image": "https://rukminim2.flixcart.com/image/850/1000/xif0q/shoe/t/3/n/3-875-mlt-3-deals4you-multi-original-imagg27xn9hnxw6h.jpeg?q=90&crop=false",
        "description": "Comfortable Running Sneakers",
        "sizes": [7, 9, 11]
        }]


app = Flask(__name__)
app.secret_key = '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'
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

SCOPES = ['https://www.googleapis.com/auth/calendar']

landmarks_global = []
isBbx = True
scale = 0

orange_image = np.ones((256, 256, 3), dtype=np.uint8) * [0, 69, 255]


@app.route('/update_variable', methods=['POST'])
def update_variable():
    global isBbx
    isBbx = not isBbx
    return 'Success', 200

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html', landmarks=landmarks_global)

def gen():
    global landmarks_global

    global isBbx
    # creating our model to draw landmarks
    mpDraw = mp.solutions.drawing_utils
    # creating our model to detect our pose
    my_pose = mp.solutions.pose
    pose = my_pose.Pose()

    """Video streaming generator function."""
    # cap = cv2.VideoCapture(0)
    while True:
        # success, img = cap.read()
        img = cv2.flip(orange_image, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(imgRGB)

        # Define the frame for the user to stand in initially
        left_corner = (100, 100)
        right_corner = (540, 540)
        # Draw corners and box
        if isBbx:

            cv2.rectangle(img, left_corner, right_corner, (0, 255, 0), 2)
            
        landmarks_list = []
        if result.pose_landmarks:
            # mpDraw.draw_landmarks(img, result.pose_landmarks, my_pose.POSE_CONNECTIONS)
            # Extract landmarks and add to landmarks_list
            for landmark in result.pose_landmarks.landmark:
                landmarks_list.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility,
                })

            # Check if the user is in the box
            if landmarks_list:
                height, width, _ = img.shape
                left_shoulder_x = int(landmarks_list[11]['x'] * width)
                right_shoulder_x = int(landmarks_list[12]['x'] * width)
                left_shoulder_y = int(landmarks_list[11]['y'] * height)
                right_shoulder_y = int(landmarks_list[12]['y'] * height)
                # left_shoulder_z = int(landmarks_list[11]['z'] )
                # right_shoulder_z = int(landmarks_list[12]['z'] )
                # img = cv2.circle(img, (right_shoulder_x, right_shoulder_y), 1, (0, 0, 0), thickness=50)  # Red point with a small radius and thickness
                # img = cv2.circle(img, (left_shoulder_x, left_shoulder_y), 1, (255, 255, 255), thickness=50)  # Red point with a small radius and thickness
                if isBbx:

                    if left_shoulder_x > left_corner[0] and left_shoulder_x < right_corner[0] and right_shoulder_y > left_corner[1] and right_shoulder_x > left_corner[0] and right_shoulder_x < right_corner[0] and left_shoulder_y > left_corner[1]:
                        cv2.putText(img, 'Click Done', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        shoulder_width = landmarks_list[11]['x'] - landmarks_list[12]['x']
                        global scale 
                        scale = shoulder_width

                    else:
                        cv2.putText(img, 'Please stand inside the box', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Update global landmarks list
        landmarks_global = landmarks_list
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(20)
        if key == 27:
            break

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_landmarks')
def get_landmarks():
    """Endpoint to get landmarks data in JSON format."""
    return jsonify(landmarks_global)

@app.route('/get_Scale')
def get_Scale():
    """Endpoint to get scale factor for the model."""
    # print(scale, "sca")
    
    return jsonify(scale= scale)

@app.route('/capture_frame', methods=['POST'])
def capture_frame():
    global orange_image
    data = request.json
    image_data = data['image']

    if image_data.startswith('data:image/jpeg;base64,'):
        image_data = image_data.replace('data:image/jpeg;base64,', '')
    elif image_data.startswith('data:image/png;base64,'):
        image_data = image_data.replace('data:image/png;base64,', '')
    else:
        return jsonify({'status': 'error', 'message': 'Unsupported image format'}), 400

    image_bytes = base64.b64decode(image_data)
    
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    img_td = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  

    orange_image = img_td
   
    return jsonify({'status': 'success'})

@app.route('/product_list')
def product_list():
    """Product list page."""
   
    return render_template('product_list.html', products=products)

@app.route('/product/<int:product_id>')
def product_details(product_id):
    """Product details page."""
    
    product = next((item for item in products if item["id"] == product_id), None)
    if product:
        return render_template('product_details.html', product=product)
    else:
        return "Product not found", 404
# ar try on
@app.route('/product/try_on_ar<string:model_name>')
def try_on_ar(model_name):
    """Try-on AR page."""
    return render_template('try_on_ar.html', model_name=model_name)
#2d try on 
@app.route('/tryon')
def try_on():
    return redirect(url_for('twod_tryon'))

@app.route('/twodtryon')
def twod_tryon():
    return render_template('twodtryon.html')


@app.template_filter('enumerate')
def enumerate_filter(s):
    return enumerate(s)   

if __name__ == '__main__':
    app.run(debug=True)