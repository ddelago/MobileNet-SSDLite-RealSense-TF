import sys
import os
import numpy as np
import cv2
import pyrealsense2 as rs
import tensorflow as tf
sys.path.append('..')
from utils import label_map_util
import visualization_utils as vis_util
from pyimagesearch.centroidtracker import CentroidTracker

def find_device_that_supports_advanced_mode() :
    ctx = rs.context()
    ds5_dev = rs.device()
    devices = ctx.query_devices()
    DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07"]
    for dev in devices:
        if dev.supports(rs.camera_info.product_id) and str(dev.get_info(rs.camera_info.product_id)) in DS5_product_ids:
            if dev.supports(rs.camera_info.name):
                print("Found device that supports advanced mode:", dev.get_info(rs.camera_info.name))
            return dev
    raise Exception("No device that supports advanced mode was found")

pipeline = None
try:
    HOME_PATH = os.path.expanduser('~')
    CWD_PATH = 'model'
    #MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
    PATH_TO_CKPT = os.path.join(CWD_PATH, 'frozen_inference_graph.pb')
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'mscoco_label_map.pbtxt')
    swapRB = True
    NUM_CLASSES = 90

    # Load the label map.
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.compat.v1.Session(graph=detection_graph)

    # Input tensor is the image
    # Output tensors are the detection boxes, scores, and classes
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Configure depth and color streams RealSense D435
    pipeline = rs.pipeline()
    config = rs.config()
    colorizer = rs.colorizer()

    with open('config.json', 'r') as file:
        json_text = file.read().strip()
        rs.rs400_advanced_mode(find_device_that_supports_advanced_mode()).load_json(json_text)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
except:
    import traceback
    traceback.print_exc()
    # Stop streaming
    if pipeline != None:
        pipeline.stop()
    print("\n\nFinished\n\n")
    sys.exit()

""" Find objects, add bounding box and class name, using vis_utils module """
def camThread():
    img, colorized_depth, depth_frame, (height, width), (boxes, scores, classes, num) = get_frame_data()

    # Draw the results of the detection (aka 'visulaize the results')
    img = vis_util.visualize_boxes_and_labels_on_image_array(
        color_image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2,
        min_score_thresh=0.55,
        depth_frame=depth_frame,
        height=height,
        width=width)

    return img, colorized_depth

""" Find objects, add bounding box and class name """
def camThreadSimple():
    img, colorized_depth, depth_frame, (height, width), (boxes, scores, classes, num) = get_frame_data()
        
    # loop over the detections
    for i in range(len(boxes[0])):
        if scores[0][i] > 0.55:
            box = tuple(boxes[0][i].tolist())
            ymin=round(box[0]*height)
            xmin=round(box[1]*width)
            ymax=round(box[2]*height)
            xmax=round(box[3]*width)

            center_x = xmin + round((xmax-xmin)/2)
            center_y = ymin + round((ymax-ymin)/2)

            depth = calc_depth(depth_frame, center_x, center_y)
            text = "{}, {:.2f} meters away".format(category_index[classes[0][i]]['name'], depth)
            
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(img, text, (center_x-10, center_y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(img, (center_x, center_y), 4, (0, 255, 0), -1)

    return img, colorized_depth

""" Find objects, assign IDs, and calculate center by reducing error """
ct = CentroidTracker()  
objects = None
def camThreadCentroid():
    img, colorized_depth, depth_frame, (height, width), (boxes, scores, classes, num) = get_frame_data()
        
    rects = []
    # loop over the detections
    for i in range(len(boxes[0])):
        if scores[0][i] > 0.55:
            box = tuple(boxes[0][i].tolist())
            ymin=round(box[0]*height)
            xmin=round(box[1]*width)
            ymax=round(box[2]*height)
            xmax=round(box[3]*width)

            center_x = xmin + round((xmax-xmin)/2)
            center_y = ymin + round((ymax-ymin)/2)
            
            rects.append((ymin, xmin, ymax, ymax))
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # update our centroid tracker using the computed set of bounding box rectangles
            objects = ct.update(rects)
            
            # loop over the tracked objects
            for (objectID, centroid) in objects.items():
                depth = calc_depth(depth_frame, centroid[0], centroid[1])
                text = "ID {}, {:.2f} meters away".format(objectID, depth)

                cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    return img, colorized_depth

def get_frame_data():
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        return
        
    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    height = color_image.shape[0]
    width = color_image.shape[1]

    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

    frame_expanded = np.expand_dims(color_image, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    return color_image, colorized_depth, depth_frame, (height, width), (boxes, scores, classes, num) 

def calc_depth(depth_frame, x, y):
    # Get an average of pixel depths
    meters = 0
    pixel_counter = 0
    for x in range(x-5, x+5) :
        for y in range(y-5, y+5) :
            meters += depth_frame.as_depth_frame().get_distance(x,y)
            pixel_counter += 1
    
    return meters/pixel_counter
    
try:
    cv2.namedWindow('Objects',cv2.WINDOW_NORMAL)
    while True:
        img, depth = camThreadSimple()
        cv2.imshow('Objects', cv2.hconcat([img, depth]))
        # Exit at the end of the video on the 'q' keypress
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except:
    import traceback
    traceback.print_exc()
finally:
    # Stop streaming
    if pipeline != None:
        pipeline.stop()
    print("\n\nFinished\n\n")
    sys.exit()

