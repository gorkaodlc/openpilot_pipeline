import matplotlib.pyplot as plt
import json
import numpy as np
import onnxruntime
import pandas as pd
import cv2
#import time
import pickle as pkl

X_IDXS = np.array([ 0. ,   0.1875,   0.75  ,   1.6875,   3.    ,   4.6875,
         6.75  ,   9.1875,  12.    ,  15.1875,  18.75  ,  22.6875,
        27.    ,  31.6875,  36.75  ,  42.1875,  48.    ,  54.1875,
        60.75  ,  67.6875,  75.    ,  82.6875,  90.75  ,  99.1875,
       108.    , 117.1875, 126.75  , 136.6875, 147.    , 157.6875,
       168.75  , 180.1875, 192.])

def parse_image(frame):
    H = (frame.shape[0]*2)//3
    W = frame.shape[1]
    parsed = np.zeros((6, H//2, W//2), dtype=np.uint8)

    parsed[0] = frame[0:H:2, 0::2]
    parsed[1] = frame[1:H:2, 0::2]
    parsed[2] = frame[0:H:2, 1::2]
    parsed[3] = frame[1:H:2, 1::2]
    parsed[4] = frame[H:H+H//4].reshape((-1, H//2,W//2))
    parsed[5] = frame[H+H//4:H+H//2].reshape((-1, H//2,W//2))

    return parsed

def seperate_points_and_std_values(df):
    points = df.iloc[lambda x: x.index % 2 == 0]
    std = df.iloc[lambda x: x.index % 2 != 0]
    points = pd.concat([points], ignore_index = True)
    std = pd.concat([std], ignore_index = True)

    return points, std


def proces_frames(path, num_frames = 100, plot = False):
    print(f'Procesing frames of {path}')
    cap = cv2.VideoCapture(path)
    parsed_images = []
    rgb_images = []
    rgb_rectangle_images = []

    #start_point = np.array([110, 410]) 
    #size = np.array([900, 450])
    start_point = np.array([170, 455]) 
    size = np.array([800, 400])
    end_point = start_point + size
    i_dx = 0
    if plot:
        plt.ion()  # Modo interactivo
        fig, ax = plt.subplots()

    while (cap.isOpened() and i_dx < num_frames):
        i_dx += 1
        ret, frame = cap.read()
        if (ret == False):
            break
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_images.append(image_rgb)

        image_crop = frame[start_point[1]:end_point[1], start_point[0]:end_point[0]]
        image_resize_crop = cv2.resize(image_crop, (512, 256))

        image_resize_crop = cv2.cvtColor(image_resize_crop, cv2.COLOR_BGR2RGB)
        parsed_images.append((image_resize_crop))
        
        # Dibujar el rectángulo sobre la imagen
        image_with_rectangle = image_rgb.copy()
        image_with_rectangle = cv2.rectangle(image_with_rectangle, start_point, end_point, (255, 0, 0), 2)
        """
        image_with_rectangle = image_resize_crop.copy()

        height, width, _ = image_with_rectangle.shape
        center_x, center_y = width // 2, height // 2

        # Dibujar líneas centrales (vertical y horizontal)
        line_color = (0, 255, 0)  # Verde
        thickness = 2
        
        cv2.line(image_with_rectangle, (center_x, 0), (center_x, height), line_color, thickness)  # Línea vertical
        cv2.line(image_with_rectangle, (0, center_y), (width, center_y), line_color, thickness)  # Línea horizontal
        """
        rgb_rectangle_images.append(image_with_rectangle)

        if plot:
            ax.imshow(image_with_rectangle)
            ax.axis('off')  # Ocultar ejes
            plt.pause(1 / 20)  # Pausar para simular la tasa de cuadros
            ax.clear()

    if plot:
        # Cerrar el video y la ventana
        cap.release()
        plt.ioff()
        plt.close()
    print(f'Succesfully procesed {i_dx} frames')
    return rgb_images, parsed_images, rgb_rectangle_images

def main(model_path, rgb_images, croped_images, rgb_rectangle_images):
    print('Starting inference process')

    parsed_images = []
    middle_points = []
    width = 512
    height = 256
    dim = (width, height)

    plan_start_idx = 0
    plan_end_idx = 4955

    lanes_start_idx = plan_end_idx
    lanes_end_idx = lanes_start_idx + 528

    lane_lines_prob_start_idx = lanes_end_idx
    lane_lines_prob_end_idx = lane_lines_prob_start_idx + 8

    road_start_idx = lane_lines_prob_end_idx
    road_end_idx = road_start_idx + 264

    session = onnxruntime.InferenceSession(model_path, None)

    plt.ion()  # Modo interactivo
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))

    ax1.set_xlim(-5, 5)
    ax1.set_ylim(0, 100)

    # Mostrar imagen inicial
    image_inicial = cv2.imread("train/corrected_frame.png")
    img_display = ax2.imshow(image_inicial)
    ax2.set_title("Imagen en tiempo real")
    ax2.axis("off")  # Ocultar ejes

    crop_display = ax3.imshow(cv2.resize(image_inicial, (512,256)))
    ax3.set_title("Imagen dada al modelo")
    ax3.axis("off")  # Ocultar ejes

    # Activar modo interactivo
    plt.ion()

    for i_dx in range(len(rgb_images)-1):
        frame_rgb = rgb_images[i_dx]
        frame_rectangulo_rgb = rgb_rectangle_images[i_dx]
        frame_crop = croped_images[i_dx]

        if frame_crop is not None:
            if frame_crop.shape[:2] != dim:
                frame_crop = cv2.resize(frame_crop, dim)
            frame_yuv = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2YUV_I420)
            parsed = parse_image(frame_yuv)

        if (len(parsed_images) >= 2):
            del parsed_images[0]

        parsed_images.append(parsed)

        if (len(parsed_images) >= 2):
            parsed_arr = np.array(parsed_images)
            parsed_arr.resize((1, 12, 128, 256))

            data = json.dumps({'data': parsed_arr.tolist()})
            data = np.array(json.loads(data)['data']).astype('float32')

            input_imgs = session.get_inputs()[0].name
            desire = session.get_inputs()[1].name
            initial_state = session.get_inputs()[2].name
            traffic_convention = session.get_inputs()[3].name
            output_name = session.get_outputs()[0].name

            desire_data = np.array([0]).astype('float32')
            desire_data.resize((1, 8))

            traffic_convention_data = np.array([0]).astype('float32')
            traffic_convention_data.resize((1, 512))

            initial_state_data = np.array([0]).astype('float32')
            initial_state_data.resize((1, 2))

            result = session.run([output_name], {input_imgs: data,
                                                 desire: desire_data,
                                                 traffic_convention: traffic_convention_data,
                                                 initial_state: initial_state_data
                                                 })

            res = np.array(result)

            # plan = res[:,:,plan_start_idx:plan_end_idx]
            lanes = res[:, :, lanes_start_idx:lanes_end_idx]
            # lane_lines_prob = res[:,:,lane_lines_prob_start_idx:lane_lines_prob_end_idx]
            lane_road = res[:, :, road_start_idx:road_end_idx]
            # lead = res[:,:,lead_start_idx:lead_end_idx]
            # lead_prob = res[:,:,lead_prob_start_idx:lead_prob_end_idx]
            # desire_state = res[:,:,desire_start_idx:desire_end_idx]
            # meta = res[:,:,meta_start_idx:meta_end_idx]
            # desire_pred = res[:,:,desire_pred_start_idx:desire_pred_end_idx]
            # pose = res[:,:,pose_start_idx:pose_end_idx]
            # recurrent_layer = res[:,:,rnn_start_idx:rnn_end_idx]

            lanes_flat = lanes.flatten()
            df_lanes = pd.DataFrame(lanes_flat)

            ll_t = df_lanes[0:66]
            ll_t2 = df_lanes[66:132]
            points_ll_t, std_ll_t = seperate_points_and_std_values(ll_t)
            points_ll_t2, std_ll_t2 = seperate_points_and_std_values(ll_t2)

            l_t = df_lanes[132:198]
            l_t2 = df_lanes[198:264]
            points_l_t, std_l_t = seperate_points_and_std_values(l_t)
            points_l_t2, std_l_t2 = seperate_points_and_std_values(l_t2)

            r_t = df_lanes[264:330]
            r_t2 = df_lanes[330:396]
            points_r_t, std_r_t = seperate_points_and_std_values(r_t)
            points_r_t2, std_r_t2 = seperate_points_and_std_values(r_t2)

            rr_t = df_lanes[396:462]
            rr_t2 = df_lanes[462:528]
            points_rr_t, std_rr_t = seperate_points_and_std_values(rr_t)
            points_rr_t2, std_rr_t2 = seperate_points_and_std_values(rr_t2)

            road_flat = lane_road.flatten()
            df_road = pd.DataFrame(road_flat)

            roadr_t = df_road[0:66]
            roadr_t2 = df_road[66:132]
            points_road_t, std_ll_t = seperate_points_and_std_values(roadr_t)
            points_road_t2, std_ll_t2 = seperate_points_and_std_values(roadr_t2)

            roadl_t = df_road[132:198]
            roadl_t2 = df_road[198:264]
            points_roadl_t, std_rl_t = seperate_points_and_std_values(roadl_t)
            points_roadl_t2, std_rl_t2 = seperate_points_and_std_values(roadl_t2)

            middle = points_ll_t2.add(points_l_t, fill_value=0) / 2
            middle_points.append(middle)
            # 🔴 BORRAR SOLO LOS ELEMENTOS DEL GRÁFICO
            ax1.cla()

            # 🔵 REESTABLECER LÍMITES Y TÍTULO
            ax1.set_xlim(-5, 5)
            ax1.set_ylim(0, 100)
            ax1.set_xlabel("Desviación lateral (m)")
            ax1.set_ylabel("Distancia (m)")
            ax1.set_title("Detección de Carril")

            # 🟢 VOLVER A DIBUJAR LOS DATOS
            ax1.plot(middle, X_IDXS, 'g', label="Middle path")
            ax1.plot(points_ll_t2, X_IDXS, 'y', label="Lane border")
            ax1.plot(points_l_t2, X_IDXS, 'y')
            ax1.plot(points_road_t, X_IDXS, 'r', label="Road border")
            ax1.plot(points_road_t2, X_IDXS, 'r')

            ax1.legend()



            #ax1.scatter(middle, X_IDXS, color="g")

            # 			plt.scatter(points_ll_t, X_IDXS, color = "b", marker = "*")
            #ax1.scatter(points_ll_t2, X_IDXS, color="y")

            #ax1.scatter(points_l_t, X_IDXS, color="y")
            # 			plt.scatter(points_l_t2, X_IDXS, color = "y", marker = "*")

            #ax1.scatter(points_road_t, X_IDXS, color="r")
            #ax1.scatter(points_road_t2, X_IDXS, color="r")

            img_display.set_data(frame_rectangulo_rgb)
            crop_display.set_data(frame_crop)

            plt.pause(0.01)

    plt.ioff()  # Desactivar modo interactivo
    plt.close()
    print(middle_points[0].to_numpy()[:,0].shape)
    print(X_IDXS.shape)

    #with open("puntos.pkl", "wb") as archivo:
    #    pkl.dump(np.array([middle_points[0].to_numpy()[:,0], X_IDXS, np.zeros(33)]), archivo)


video_path =  'videos/corrected_video_josu.avi'
model_path = 'common/models/supercombo.onnx'
rgb_images, cropped_images, rgb_rectangle_images = proces_frames(video_path, num_frames=100, plot=False)
main(model_path, rgb_images, cropped_images, rgb_rectangle_images)
