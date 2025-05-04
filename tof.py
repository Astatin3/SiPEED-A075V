import struct
import numpy as np
import cv2
from PIL import Image

#import matplotlib.pyplot as plt
import requests

from depth import img2depth
from matchdepth import align_depth_maps

# import open3d as o3d

HOST = '192.168.233.1'
PORT = 80

def get_frame_from_http(host=HOST, port=PORT):
    r = requests.get('http://{}:{}/getdeep'.format(host, port))
    if(r.status_code == requests.codes.ok):
        # print('Get deep image')
        deepimg = r.content
        # print('Length={}'.format(len(deepimg)))
        (frameid, stamp_msec) = struct.unpack('<QQ', deepimg[0:8+8])
        # print((frameid, stamp_msec/1000))
        return deepimg

def post_encode_config(config, host=HOST, port=PORT):
    r = requests.post('http://{}:{}/set_cfg'.format(host, port), config)
    if(r.status_code == requests.codes.ok):
        return True
    return False

def frame_config_decode(frame_config):
    '''
        @frame_config bytes

        @return fields, tuple (trigger_mode, deep_mode, deep_shift, ir_mode, status_mode, status_mask, rgb_mode, rgb_res, expose_time)
    '''
    return struct.unpack("<BBBBBBBBi", frame_config)

def frame_config_encode(trigger_mode=1, deep_mode=1, deep_shift=255, ir_mode=1, status_mode=2, status_mask=7, rgb_mode=1, rgb_res=0, expose_time=0):
    '''
        @trigger_mode, deep_mode, deep_shift, ir_mode, status_mode, status_mask, rgb_mode, rgb_res, expose_time

        @return frame_config bytes
    '''
    return struct.pack("<BBBBBBBBi",
                       trigger_mode, deep_mode, deep_shift, ir_mode, status_mode, status_mask, rgb_mode, rgb_res, expose_time)

def frame_payload_decode(frame_data: bytes, with_config: tuple):
    '''
        @frame_data, bytes

        @with_config, tuple (trigger_mode, deep_mode, deep_shift, ir_mode, status_mode, status_mask, rgb_mode, rgb_res, expose_time)

        @return imgs, tuple (deepth_img, ir_img, status_img, rgb_img)
    '''
    deep_data_size, rgb_data_size = struct.unpack("<ii", frame_data[:8])
    frame_payload = frame_data[8:]
    # 0:16bit 1:8bit, resolution: 320*240
    deepth_size = (320*240*2) >> with_config[1]
    deepth_img = struct.unpack("<%us" % deepth_size, frame_payload[:deepth_size])[
        0] if 0 != deepth_size else None
    frame_payload = frame_payload[deepth_size:]

    # 0:16bit 1:8bit, resolution: 320*240
    ir_size = (320*240*2) >> with_config[3]
    ir_img = struct.unpack("<%us" % ir_size, frame_payload[:ir_size])[
        0] if 0 != ir_size else None
    frame_payload = frame_payload[ir_size:]

    status_size = (320*240//8) * (16 if 0 == with_config[4] else
                                  2 if 1 == with_config[4] else 8 if 2 == with_config[4] else 1)
    status_img = struct.unpack("<%us" % status_size, frame_payload[:status_size])[
        0] if 0 != status_size else None
    frame_payload = frame_payload[status_size:]

    assert(deep_data_size == deepth_size+ir_size+status_size)

    rgb_size = len(frame_payload)
    assert(rgb_data_size == rgb_size)
    rgb_img = struct.unpack("<%us" % rgb_size, frame_payload[:rgb_size])[
        0] if 0 != rgb_size else None

    if (not rgb_img is None):
        if (1 == with_config[6]):
            jpeg = cv2.imdecode(np.frombuffer(
                rgb_img, 'uint8', rgb_size), cv2.IMREAD_COLOR)
            if not jpeg is None:
                rgb = cv2.cvtColor(jpeg, cv2.COLOR_BGR2RGB)
                rgb_img = rgb.tobytes()
            else:
                rgb_img = None
        # elif 0 == with_config[6]:
        #     yuv = np.frombuffer(rgb_img, 'uint8', rgb_size)
        #     print(len(yuv))
        #     if not yuv is None:
        #         rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV420P2RGB)
        #         rgb_img = rgb.tobytes()
        #     else:
        #         rgb_img = None

    return (deepth_img, ir_img, status_img, rgb_img)






def scale_and_shift(image, scale_factor, shift_x, shift_y):
    """
    Scale an RGB image and shift it by n pixels in x and y direction.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input RGB image with shape (height, width, 3)
    scale_factor : float
        Scale factor (e.g., 0.5 for half size, 2.0 for double size)
    shift_x : int
        Number of pixels to shift in x direction (positive: right, negative: left)
    shift_y : int
        Number of pixels to shift in y direction (positive: down, negative: up)
    
    Returns:
    --------
    numpy.ndarray
        Scaled and shifted image with the same shape as the input image
    """
    # Get original image dimensions
    height, width = image.shape[:2]
    
    # Calculate new dimensions after scaling
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    
    # Scale the image
    scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Create a transformation matrix for the shift
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    
    # Apply the shift to the scaled image
    shifted_image = cv2.warpAffine(scaled_image, M, (new_width, new_height))
    
    # Create a blank canvas with original dimensions
    result = np.zeros_like(image)
    
    # Calculate the region to copy from the shifted_scaled image
    y_start = max(0, -shift_y)
    y_end = min(new_height, height - shift_y)
    x_start = max(0, -shift_x)
    x_end = min(new_width, width - shift_x)
    
    # Calculate the region to paste into the result image
    result_y_start = max(0, shift_y)
    result_y_end = min(height, new_height + shift_y)
    result_x_start = max(0, shift_x)
    result_x_end = min(width, new_width + shift_x)
    
    # Copy the visible part of the shifted image to the result
    if (y_end > y_start and x_end > x_start and 
        result_y_end > result_y_start and result_x_end > result_x_start):
        result[result_y_start:result_y_end, result_x_start:result_x_end] = shifted_image[y_start:y_end, x_start:x_end]
    
    return result


prev_status = None
prev_depth = None

def show_frame(frame_data: bytes):
    global prev_status
    global prev_depth
    config = frame_config_decode(frame_data[16:16+12])
    frame_bytes = frame_payload_decode(frame_data[16+12:], config)

    depth = np.frombuffer(frame_bytes[0], 'uint16' if 0 == config[1] else 'uint8').reshape(
        240, 320) if frame_bytes[0] else None

    ir = np.frombuffer(frame_bytes[1], 'uint16' if 0 == config[3] else 'uint8').reshape(
        240, 320) if frame_bytes[1] else None

    status = np.frombuffer(frame_bytes[2], 'uint16' if 0 == config[4] else 'uint8').reshape(
        240, 320) if frame_bytes[2] else None

    rgb = np.frombuffer(frame_bytes[3], 'uint8').reshape(
        (480, 640, 3) if config[6] == 1 else (600, 800, 3)) if frame_bytes[3] else None

    if not (depth is None or status is None or rgb is None):
        rgb = cv2.resize(rgb, dsize=(320, 240), interpolation=cv2.INTER_CUBIC) # Resize
        rgb = scale_and_shift(rgb, 1.1, -10, -10)
        status = 1-status

        if prev_status is None:
            mask = (status)
        else:
            mask = (status)*(prev_status)
        prev_status = status

        depth = depth*mask
        if prev_depth is not None:
            new_depth = (depth + prev_depth)/2
            prev_depth = depth
            depth = new_depth
        else:
            prev_depth = depth


        img_depth = img2depth(rgb)

        aligned_img_depth = align_depth_maps(depth, img_depth, mask)*(1-mask)

        
        return (aligned_img_depth + depth), rgb, mask
    return None


# create visualizer and window.
# vis = o3d.visualization.Visualizer()
# vis.create_window(height=480, width=640)

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(np.random.rand(10, 3))

# vis.add_geometry(pcd)


# depth_to_color_translation = np.array([0, 0, 0])  # 5cm offset in x
# depth_to_color_rotation = np.eye(3)  # Identity matrix if cameras are parallel

# color_intrinsics = (520, 520, 325, 245)

keep_running = True

photocount = 0

while keep_running:
    if post_encode_config(frame_config_encode(1,0,255,0,2,7,1,0,0)):
        p = get_frame_from_http()
        depth_image, rgb, mask = show_frame(p)
        if depth_image is None: continue
        depth_colored = cv2.applyColorMap((depth_image).astype(np.uint8), cv2.COLORMAP_JET)

        # mask = (depth_image>1000)

        # b = np.repeat((depth_image>10)[:, :, np.newaxis], 3, axis=2)
        # b = np.repeat((mask==1)[:, :, np.newaxis], 3, axis=2)


        cv2.imshow("depth", depth_colored)
        cv2.imshow("rgb", rgb)

        key = cv2.waitKey(1)

        if key & 0xFF == 27:
            break
        elif key & 0xFF == 32:
            photocount += 1
            depth = Image.fromarray(depth_image)
            rgb = Image.fromarray(rgb)

            depth.save(f"./depth/depth-{photocount}.png")
            rgb.save(f"./rgb/rgb-{photocount}.png")


            print(f"Took photo {photocount}!")

            
        
        # points, colors = depth_to_colored_points(
        #     depth_image,
        #     color_image,
        #     x_map,
        #     y_map,
        #     color_intrinsics,
        #     depth_to_color_translation,
        #     depth_to_color_rotation
        # )


        # points = depth_image.reshape((-1, 3))
        # colors = color_image.reshape((-1, 3)).astype(np.float64) / 255.0

        # colors[:, [0, 2]] = colors[:, [2, 0]]

        # print(points.shape)
        # print(colors.shape)

        # pcd.points = o3d.utility.Vector3dVector(points)
        # pcd.colors = o3d.utility.Vector3dVector(colors)


    # vis.update_geometry(pcd)

    # keep_running = vis.poll_events()
    # vis.update_renderer()

        # pcd.points.extend(np.random.rand(n_new, 3))
    # cv2.waitKey(1)
    # with open("rgbd.raw", 'wb') as f:
    #     f.write(p)
    #     f.flush()
cv2.destroyAllWindows()