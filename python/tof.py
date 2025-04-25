import struct
import numpy as np
import cv2

#import matplotlib.pyplot as plt
import requests

import open3d as o3d

HOST = '192.168.233.1'
PORT = 80

# def create_point_cloud_map(width, height, fx, fy, cx, cy):
#     """
#     Create mapping arrays for converting depth image to point cloud.
    
#     Args:
#         width, height: Image dimensions
#         fx, fy: Focal lengths
#         cx, cy: Principal point coordinates
    
#     Returns:
#         x_map, y_map: Arrays that when multiplied by depth give X,Y coordinates
#     """
#     # Create pixel coordinate grid
#     v, u = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
#     # Convert to normalized image coordinates
#     x_map = (u - cx) / fx
#     y_map = (v - cy) / fy
    
#     return x_map, y_map

# def depth_to_points(depth_image, x_map, y_map):
#     """
#     Convert depth image to point cloud using pre-computed maps.
    
#     Args:
#         depth_image: 2D depth array
#         x_map, y_map: Pre-computed coordinate maps
    
#     Returns:
#         points: Nx3 array of XYZ coordinates
#     """
#     # Calculate X and Y coordinates
#     X = depth_image * x_map
#     Y = depth_image * y_map
    
#     # Stack coordinates into point cloud
#     valid_points = depth_image > 0
#     points = np.stack((
#         X[valid_points],
#         Y[valid_points],
#         depth_image[valid_points]
#     ), axis=-1)
    
#     return points





def create_point_cloud_map(width, height, fx, fy, cx, cy):
    """
    Create mapping arrays for converting depth image to point cloud.
    
    Args:
        width, height: Image dimensions
        fx, fy: Focal lengths
        cx, cy: Principal point coordinates
    
    Returns:
        x_map, y_map: Arrays that when multiplied by depth give X,Y coordinates
    """
    # Create pixel coordinate grid
    v, u = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # Convert to normalized image coordinates
    x_map = (u - cx) / fx
    y_map = (v - cy) / fy
    
    return x_map, y_map

def transform_points(points, translation, rotation=None):
    """
    Apply rigid transformation to points.
    
    Args:
        points: Nx3 array of XYZ coordinates
        translation: [tx, ty, tz] translation vector
        rotation: 3x3 rotation matrix (optional)
    
    Returns:
        transformed_points: Nx3 array of transformed coordinates
    """
    if rotation is not None:
        points = points @ rotation.T
    return points + translation

def depth_to_colored_points(depth_image, color_image, x_map, y_map, 
                          color_intrinsics, depth_to_color_translation,
                          depth_to_color_rotation=None):
    """
    Convert depth image to colored point cloud using pre-computed maps.
    
    Args:
        depth_image: 2D depth array
        color_image: RGB image array (height, width, 3)
        x_map, y_map: Pre-computed coordinate maps for depth camera
        color_intrinsics: (fx, fy, cx, cy) for RGB camera
        depth_to_color_translation: [tx, ty, tz] from depth to color camera
        depth_to_color_rotation: 3x3 rotation matrix (optional)
    
    Returns:
        points: Nx3 array of XYZ coordinates
        colors: Nx3 array of RGB values
    """
    # Calculate initial point cloud from depth
    valid_points = depth_image > 0
    X = depth_image * x_map
    Y = depth_image * y_map
    
    points = np.stack((
        X[valid_points],
        Y[valid_points],
        depth_image[valid_points]
    ), axis=-1)
    
    # Transform points to color camera coordinate system
    transformed_points = transform_points(
        points, 
        depth_to_color_translation,
        depth_to_color_rotation
    )
    
    # Project points into color image
    fx, fy, cx, cy = color_intrinsics
    u = (transformed_points[:, 0] * fx / transformed_points[:, 2] + cx).astype(int)
    v = (transformed_points[:, 1] * fy / transformed_points[:, 2] + cy).astype(int)
    
    # Filter points that project outside image bounds
    height, width = color_image.shape[:2]
    valid_uvs = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    
    # Sample colors from valid projections
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    colors[valid_uvs] = color_image[v[valid_uvs], u[valid_uvs]]
    
    return points[valid_uvs], colors[valid_uvs]

def depth_to_points(depth_image, x_map, y_map):
    """
    Convert depth image to point cloud using pre-computed maps.
    
    Args:
        depth_image: 2D depth array
        x_map, y_map: Pre-computed coordinate maps
    
    Returns:
        points: Nx3 array of XYZ coordinates
    """
    # Calculate X and Y coordinates
    X = depth_image * x_map
    Y = depth_image * y_map
    
    # Stack coordinates into point cloud
    valid_points = depth_image > 0
    points = np.stack((
        X[valid_points],
        Y[valid_points],
        depth_image[valid_points]
    ), axis=-1)
    
    return points







def get_frame_from_http(host=HOST, port=PORT):
    r = requests.get('http://{}:{}/getdeep'.format(host, port))
    if(r.status_code == requests.codes.ok):
        print('Get deep image')
        deepimg = r.content
        print('Length={}'.format(len(deepimg)))
        (frameid, stamp_msec) = struct.unpack('<QQ', deepimg[0:8+8])
        print((frameid, stamp_msec/1000))
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

prev_status = None

def show_frame(frame_data: bytes):
    global prev_status
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
        if prev_status is None:
            mask = (status==0)
        else:
            mask = (status==0)*(prev_status==0)
        
        linear_mask = mask.reshape((-1))
        # delete = np.where(1-linear_mask))

        # depth_frame = (depth*mask)/1000
        depth_frame = (depth)/1000
        blurred = cv2.GaussianBlur(depth_frame,(3,3),1.0)
        # depth_frame = cv2.addWeighted(depth_frame, 2.5, blurred, -1, 0)

        prev_status = status
        points = depth_to_points(depth_frame, x_map, y_map)
        points = points[linear_mask]

        colors = cv2.resize(rgb, (320, 240))
        # cv2.imshow("color", mask)

         
        
        # colors = (np.stack((depth_frame,) * 3, axis=-1)).reshape((-1, 3)).astype(np.float64) / 255.0
        colors = colors.reshape((-1, 3)).astype(np.float64) / 255.0
        # colors *= linear_mask
        # colors = colors[:-(colors.shape[0]-points.shape[0])]
        # colors = np.delete(colors, delete, axis = 0)
        colors = colors[linear_mask]

        # print(np.where(points))

        # print(colors_e.shape)


        
        return depth_frame, points, colors
    return None, None, None


# create visualizer and window.
vis = o3d.visualization.Visualizer()
vis.create_window(height=480, width=640)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.random.rand(10, 3))

vis.add_geometry(pcd)

x_map, y_map = create_point_cloud_map(
    width=320, height=240,
    fx=231.8290, fy=232.7785,  # focal lengths
    cx=166.9372, cy=123.5151   # principal point
)


depth_to_color_translation = np.array([0, 0, 0])  # 5cm offset in x
depth_to_color_rotation = np.eye(3)  # Identity matrix if cameras are parallel

color_intrinsics = (520, 520, 325, 245)

keep_running = True

while keep_running:
    if post_encode_config(frame_config_encode(1,0,255,0,2,7,1,0,0)):
        p = get_frame_from_http()
        depth_image, points, colors = show_frame(p)
        if depth_image is None or colors is None : continue
        cv2.imshow("e", depth_image)
        cv2.waitKey(1)

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

        print(points.shape)
        print(colors.shape)

        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)


    vis.update_geometry(pcd)

    keep_running = vis.poll_events()
    vis.update_renderer()

        # pcd.points.extend(np.random.rand(n_new, 3))
    # cv2.waitKey(1)
    # with open("rgbd.raw", 'wb') as f:
    #     f.write(p)
    #     f.flush()
