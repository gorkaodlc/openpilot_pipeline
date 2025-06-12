import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from train.dataloader import load_transformed_video
from utils import Calibration, draw_path, FULL_FRAME_SIZE, extract_preds, transform_img, transform_frames, medmodel_intrinsics
from matplotlib.patches import Patch
from train.model import load_inference_model
import cv2
from utils import bgr_to_yuv, reshape_yuv, create_image_canvas
from common.config import eon_focal_length, FOCAL, FULL_FRAME_SIZE, W, H, eon_intrinsics, rvec, tvec, CAMARA_HEIGHT, CAR_WIDTH
import time 

""" Clase que carga el modelo y tiene funciones para la inferencia en tiempo real del modelo supercombo
 - Constructor
    + Carga el modelo supercombo
    + Carga todas las funciones necesarias para la inferencia
    + Inicializar todas las variables (deseo, trafico y asi) para como los quiere el modelo
 - Método que recarga el modelo, el contructor llama a este
 - Metodo que corre el modelo
"""

X_IDXs = [
        0.,   0.1875,   0.75,   1.6875,   3.,   4.6875,
        6.75,   9.1875,  12.,  15.1875,  18.75,  22.6875,
        27.,  31.6875,  36.75,  42.1875,  48.,  54.1875,
        60.75,  67.6875,  75.,  82.6875,  90.75,  99.1875,
        108., 117.1875, 126.75, 136.6875, 147., 157.6875,
        168.75, 180.1875, 192.]

plot_img_width, plot_img_height = 640, 480 

fill_color_gt = [0,  255, 0]
line_color_gt = [255,255, 0]
fill_color_preds = [0,  0,255]
line_color_preds = [200,0,255]
laneline_colors = [(255, 0, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255)]

def legend_color(color, opacity=0.3):
    return np.append(np.array(color)/255, opacity)

zoom_x = FULL_FRAME_SIZE[0] / plot_img_width
zoom_y = FULL_FRAME_SIZE[1] / plot_img_height

CALIB_BB_TO_FULL = np.asarray([
    [zoom_x, 0., 0.],
    [0., zoom_y, 0.],
    [0., 0., 1.]])

PLOT_WIDTH = 12

class SuperComboRT:
    def __init__(self, model_path, project_path):
        self.model_path = model_path
        self.project_path = project_path
        
        # Cargar el modelo por primera vez
        self.load_model(self.model_path)
            
        # Inicializar variables de las predicciones        
        self.lanelines_preds = []
        self.road_edges_preds = []
        self.path_preds = []
        
        rpy_calib_pred = np.array([0.0, -0.018, 0.0])
        #rpy_calib_pred = np.array([0.0, 0.0, 0.0])
        self.calibration = Calibration(rpy_calib_pred, plot_img_width=plot_img_width, plot_img_height=plot_img_height)

        
    def load_model(self, model_path):
        # Cargar modelo supercombo
        self.model, self.run_model = load_inference_model(model_path)
        
        # Inicializar las variables usadas por el modelo
        self.frames_input = np.zeros((1, 12, 128, 256), dtype=np.uint8)
        self.last_processed_frame = np.zeros((1, 6, 128, 256), dtype=np.uint8)
        self.frame_idx = 0
        
        self.desire_data = np.array([0]).astype('float32')
        self.desire_data.resize((1,8))

        self.initial_state_data = np.array([0]).astype('float32')
        self.initial_state_data.resize((1,2))

        self.recurrent_state = np.zeros((1, 512), dtype=np.float32)
        
        self.traffic_convetion = np.array([0, 1], dtype=np.float32).reshape(1, 2) # Carril derecho
        
        self.model_inputs = {
            'input_imgs': self.frames_input,
            'desire': self.desire_data,
            'traffic_convention': self.traffic_convetion,
            'initial_state': self.recurrent_state
        }
        
        self.outs = np.zeros((1, 6409), dtype=np.float32)
    
    def run_model_frame(self, frame):
        self.last_frame = frame
        self.frame_idx += 1
        
        # Transformar frame como lo quiere el modelo
        yuv_frame = bgr_to_yuv(self.last_frame)
        img_med_model = transform_img(yuv_frame, 
                                        augment_eulers=rvec,
                                        augment_trans=tvec,
                                        from_intr=eon_intrinsics,
                                        to_intr=medmodel_intrinsics, 
                                        top_hacks=False,
                                        yuv=True,
                                        output_size=(512, 256))


        # Formatear frames para el modelo
        self.frames_input[0, :6] = self.last_processed_frame[0]
        
        # Guardar ultimo frame procesado en formato numpy
        self.last_processed_frame[0] = reshape_yuv(img_med_model.reshape(1, img_med_model.shape[0], img_med_model.shape[1]))        
        self.frames_input[0, 6:] = self.last_processed_frame[0]
        
        # Gestion de primer frame
        if self.frame_idx <= 1: 
            self.frames_input[0, :6] = self.last_processed_frame[0]                
    
        # Formatear inputs
        self.model_inputs['input_imgs'] = self.frames_input.astype(np.float32)
        self.model_inputs['initial_state'] = self.recurrent_state
        
        # Correr modelo
        self.outs, self.recurrent_state = self.run_model(self.model_inputs)
                
        # Procesar outputs
        lanelines, road_edges, best_path = extract_preds(self.outs, best_plan_only=True)[0]
        
        self.lanelines_preds.append(lanelines)
        self.road_edges_preds.append(road_edges)
        self.path_preds.append(best_path)
        
        # Generar imagen rgb para proyección
        if self.project_path:            
            self.generate_img_last_prediction()
        
        # Devolver las predicciones
        return lanelines, road_edges, best_path

    def generate_img_last_prediction(self, camara_height = CAMARA_HEIGHT, car_width = CAR_WIDTH):
        rgb_frame =  cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
        self.last_rgb_frame = create_image_canvas(rgb_frame, CALIB_BB_TO_FULL, plot_img_height, plot_img_width)      

        self.img_plot = draw_path(self.lanelines_preds[-1], self.road_edges_preds[-1], 
                             self.path_preds[-1][0, :, :3], self.last_rgb_frame, self.calibration, laneline_colors, fill_color=fill_color_preds, line_color=line_color_preds, height=camara_height, width=car_width)    

        return self.img_plot
    
if __name__ == "__main__":
    ORIGINAL_MODEL = '/home/VICOMTECH/gortiz/Proyectos/Mendi4U/openpilot-pipeline/common/models/supercombo.onnx'
    #PATH_TO_SEGMENT = '/home/VICOMTECH/gortiz/Proyectos/Mendi4U/supercombo_evaluation/records/20250314_102322_RecFile_1/RecFile_1_20250314_102322_undistort_1_1_imageOut.avi'    
    PATH_TO_SEGMENT = '/home/VICOMTECH/gortiz/Proyectos/Mendi4U/openpilot-pipeline/videos/video_gps.avi'
    #PATH_TO_SEGMENT = '/home/VICOMTECH/gortiz/Proyectos/Mendi4U/openpilot-pipeline/videos/recta-1.avi'    
    #PATH_TO_SEGMENT = '/home/VICOMTECH/gortiz/Proyectos/Mendi4U/supercombo_evaluation/records/20240228_104551_RecFile_1_1_Amezketa_Abaltzisketa/RecFile_1_1_20240228_104551_Video4Linux_1_output.avi'
    modeloSC = SuperComboRT(ORIGINAL_MODEL, True)
    
    segment_video = cv2.VideoCapture(PATH_TO_SEGMENT)

    fig, ax = plt.subplots()
    while segment_video.isOpened():        
        t0 = time.time()
        ret, frame = segment_video.read()
        print(frame.shape)
        modeloSC.run_model_frame(frame)
        rgb_frame = modeloSC.img_plot
        t1 = time.time()
        
        ax.clear()
        ax.imshow(rgb_frame)
        
        ax.set_title(f'Frame {modeloSC.frame_idx}, {round(t1-t0, 4)} seg')
        t2 = time.time()
        wait_time = 0.05 - (t2 - t0)
        #if wait_time > 0.0:
        plt.pause(0.001)
        