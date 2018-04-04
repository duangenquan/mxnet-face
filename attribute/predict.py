import os, time
import argparse,logging
import numpy as np
import mxnet as mx
import cv2, dlib, urllib
from lightened_moon import lightened_moon_feature
from shutil import copyfile

logger = logging.getLogger()
logger.setLevel(logging.INFO)
import pdb


def detectFaces(img, detector):
    faces = detector(img, 1)
    return faces
    
def detectFacesInCascade(img, face_cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    opencv_faces = face_cascade.detectMultiScale(gray, 1.3, 5, 0, (20,20))
    
    faces = []
    for (x,y,w,h) in opencv_faces:
    #print x, y, x+w, y+h
        faces.append(dlib.rectangle(int(x), int(y), int(x+w), int(y+h)))
        
    return faces

def attribute(img, detector, face_cascade, devs, symbol, arg_params, aux_params, argssize):
    h,w,c=img.shape
    scale = max(h/240.0, w/320.0)
    neww = int(w/scale)
    newh = int(h/scale)
    newimg = cv2.resize(img,(neww, newh))
        
    # read img and drat face rect
    faces = detectFaces(newimg, detector)
    if len(faces) == 0:
        faces = detectFacesInCascade(newimg, face_cascade)
    
    gray = np.zeros(img.shape[0:2])

            
    for i in range(len(faces)):
        faces[i] = dlib.rectangle(int(faces[i].left() *scale), int(faces[i].top()*scale), int(faces[i].right()*scale), int(faces[i].bottom()*scale))
            
    font = cv2.FONT_HERSHEY_SIMPLEX
    if len(faces) > 0:
            max_face = max(faces, key=lambda rect: rect.width() * rect.height())
    
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            pad = [0.25, 0.25, 0.25, 0.25]
            left = int(max(0, max_face.left() - max_face.width()*float(pad[0])))
            top = int(max(0, max_face.top() - max_face.height()*float(pad[1])))
            right = int(min(gray.shape[1], max_face.right() + max_face.width()*float(pad[2])))
            bottom = int(min(gray.shape[0], max_face.bottom()+max_face.height()*float(pad[3])))
            
            gray = gray[top:bottom, left:right]
            #print gray.shape, left, right, top, right,  argssize
            gray = cv2.resize(gray, (argssize, argssize))/255.0
            imgpred= np.expand_dims(np.expand_dims(gray, axis=0), axis=0)
            # get pred
            arg_params['data'] = mx.nd.array(imgpred, devs)
            exector = symbol.bind(devs, arg_params ,args_grad=None, grad_req="null", aux_states=aux_params)
            exector.forward(is_train=False)
            exector.outputs[0].wait_to_read()
            output = exector.outputs[0].asnumpy()
            text = ["5_o_Clock_Shadow","Arched_Eyebrows","Attractive","Bags_Under_Eyes","Bald", "Bangs","Big_Lips","Big_Nose",
            "Black_Hair","Blond_Hair","Blurry","Brown_Hair","Bushy_Eyebrows","Chubby","Double_Chin","Eyeglasses","Goatee",
            "Gray_Hair", "Heavy_Makeup","High_Cheekbones","Male","Mouth_Slightly_Open","Mustache","Narrow_Eyes","No_Beard",
            "Oval_Face","Pale_Skin","Pointy_Nose","Receding_Hairline","Rosy_Cheeks","Sideburns","Smiling","Straight_Hair",
            "Wavy_Hair","Wearing_Earrings","Wearing_Hat","Wearing_Lipstick","Wearing_Necklace","Wearing_Necktie","Young"]
            pred = np.ones(40)
            #print("attribution is:")
            cnt = 0 
            for i in range(40):
                #print text[i].rjust(20)+" : \t",
                if output[0][i] < 0:
                    pred[i] = -1
                    #print "No"
                else:
                    pred[i] = 1
                    #print "Yes"
                    cv2.putText(img, text[i], (20, 50 + cnt * 30), font, 1, (255,255,0), 2, cv2.LINE_AA) 
                    cnt = cnt + 1
    else:
        cv2.putText(img, "No detected faces", (20,  50), font, 1, (255,255,0), 2, cv2.LINE_AA)

    for f in faces:
        if f == max_face:
            cv2.rectangle(img, (f.left(), f.top()), (f.right(), f.bottom()), (0,0,255), 2)
        else:
            cv2.rectangle(img, (f.left(), f.top()), (f.right(), f.bottom()), (255,0,0), 2)

    return img

def main(args):
    symbol = lightened_moon_feature(num_classes=40, use_fuse=True)
    devs = None
    if args.gpus is not None:
        print("use gpu...")
        devs = mx.gpu()
    else:
        print("use cpu...")
        devs = mx.cpu()
    _, arg_params, aux_params = mx.model.load_checkpoint(args.model_load_prefix, args.model_load_epoch)
    detector = dlib.get_frontal_face_detector()
    face_cascade = cv2.CascadeClassifier(args.opencv)

    stream = urllib.urlopen(args.url)
    bytes = ''
    temp_count = 16384

    while True:

        #Get a new frame
        imbytes = None
        temp_bytes = stream.read(temp_count)
        if temp_bytes:
            bytes+=temp_bytes
        else:
            stream = urllib.urlopen(args.url)
            bytes = ''
            temp_count = 16384

        a = bytes.find('\xff\xd8')
        b = bytes.find('\xff\xd9')
        if a != -1 and b != -1:

            jpg = bytes[a:b + 2]
            temp_count = b + 2 - a
            bytes = bytes[b + 2:]
            imbytes = jpg

        if imbytes is None:
            continue

        frame = cv2.imdecode(np.fromstring(imbytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        result = attribute(frame, detector, face_cascade, devs, symbol, arg_params, aux_params, args.size)
        cv2.imshow("demo", result)
        cv2.waitKey(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="predict the face attribution of one input image")
    parser.add_argument('--gpus', type=str, help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--img', type=str, default='./test.jpg', help='the input img path')
    parser.add_argument('--url', type=str, default='', help='the input stream url')
    parser.add_argument('--size', type=int, default=128,
                        help='the image size of lfw aligned image, only support squre size')
    parser.add_argument('--opencv', type=str, default='../model/opencv/cascade.xml',
                        help='the opencv model path')
    parser.add_argument('--pad', type=float, nargs='+',
                                 help="pad (left,top,right,bottom) for face detection region")
    parser.add_argument('--model-load-prefix', type=str, default='../model/lightened_moon/lightened_moon_fuse',
                        help='the prefix of the model to load')
    parser.add_argument('--model-load-epoch', type=int, default=82,
                        help='load the model on an epoch using the model-load-prefix')
    args = parser.parse_args()
    logging.info(args)
    main(args)

