from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import time
import datetime
import argparse
import json
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    

    
    def detect_image(images):

        input_imgs = Variable(images.type(Tensor))

        # Get detections
        with torch.no_grad():
            try:
                detections = model(input_imgs)
            except:
                pass
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        img_detections.extend(detections)
    

        image_batch = []
        res_batch = []
        for n, image in enumerate(images):
            font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness = (image.size[0] + image.size[1]) // 300
        
            
            res = []
            for i, x1, y1, x2, y2, conf, cls_conf, c in enumerate(img_detections[n]):
                predicted_class = classes[c]
                box = y1, x1, y2, x2
                score = conf
        
                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
        
                top, left, bottom, right = box
                
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        #            print(label, (left, top), (right, bottom))
                res.append([label,left,top,right,bottom,(left + right)/2,(top + bottom)/2])
        
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])
        
                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                del draw
            image_batch.append(image)
        
            res_batch.append(res)
        end = timer()
        #        print(end - start)
        return image_batch,res_batch  





    
    
    
    
    
        # Bounding-box colors
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    
        print("\nSaving images:")
        # Iterate through images and save plot of detections
    
        ################################################    
        accu = 0
        with open('res.json',encoding='utf-8') as f:
            a = f.readlines()[0]
    #                info_str = re.sub('\n','',''.join(f.readlines()))
    #                print(info_str)
            info = json.loads(a)    
        ###############################################
        for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
    
            print("(%d) Image: '%s'" % (img_i, path))
    
            # Create plot
            img = np.array(Image.open(path))
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(img)
    
            ##########################
            flags = 0  ###有人没戴就是1
            ##########################
            # Draw bounding boxes and labels of detections
            if detections is not None:
                
                # Rescale boxes to original image
                detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)
                
                ###########################
                persons = []
                helmets = []
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    if cls_pred == 0:
    #                    print(x1,y1,x2,y2)
                        helmets.append([x1, y1, x2, y2, conf, cls_conf, cls_pred])
                    else:
    #                    print(x1,y1,x2,y2)
                        persons.append([x1, y1, x2, y2, conf, cls_conf, cls_pred])
                    
                
                for person in persons:
                    if person[0]<10 or person[1]<10 or person[2]>1270 or person[3]>710:
                        continue
                    y_down_thres = person[1] + (person[3] - person[1])/3*2
                    y_up_thres = person[1] - (person[3] - person[1])/3*2
                    x_right_thres = person[2] + (person[2] - person[0])/3*2
                    x_left_thres = person[0] - (person[2] - person[0])/3*2
                    flag = 1
                    for helmet in helmets:
                        print((helmet[0],x_left_thres), (helmet[1],y_up_thres),(helmet[2],x_right_thres),(helmet[3],y_down_thres))
                        if helmet[0]>x_left_thres and helmet[1]>y_up_thres and \
                            helmet[2]<x_right_thres and helmet[3]<y_down_thres:
                            flag = 0
                    flags += flag
                ###########################
            
            
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
    
                    print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
    
                    box_w = x2 - x1
                    box_h = y2 - y1
    
                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(
                        x1,
                        y1,
                        s=classes[int(cls_pred)],
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0},
                    )
            ################################
            print('%s:%s' %(path,flags))
                
            if flags > 0:
                flags = 1
    
            print(info[os.path.basename(path)])
            if int(info[os.path.basename(path)]) == flags:
                accu += 1 
            
            print(accu,img_i+1)
            
            print(accu/(img_i+1))
            ##############################
    
            # Save generated image with detections
            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            filename = path.split("/")[-1][:-4]
            plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
            plt.close()
            
            
        print(accu/(img_i+1))
        
        
    
    
    
    
  
    