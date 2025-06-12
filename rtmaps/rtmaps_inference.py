import rtmaps.core as rt
import rtmaps.types
from rtmaps.base_component import BaseComponent  # base class
import numpy as np
import math
from matplotlib import pyplot as plt
import cv2
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference.openpilot_realtime_inf import SuperComboRT
from PIL import Image
from PIL import ImageDraw
import utm
import scipy.interpolate as interp
import json 
import time

def calculate_trajectory_error(predicted_points, ground_truth_points, thresholds=(0.5, 1, 2)):
    """
    Calcula la distancia euclídea entre los puntos predichos y la interpolación de los puntos del ground truth.
    También calcula la métrica AP para diferentes umbrales.
    
    Parámetros:
    - predicted_points: np.array de forma (N,2) con los puntos predichos (x, y)
    - ground_truth_points: np.array de forma (M,2) con los puntos del ground truth (x, y), equidistantes
    - thresholds: lista o tupla de valores umbral para calcular AP
    
    Devuelve:
    - errores por rango de distancia en Y
    - mean_error: error medio total
    - max_error: error máximo total
    - std_error: desviación estándar del error total
    - hit_rates: diccionario con los AP@threshold para cada umbral
    - errors: lista de errores punto a punto
    """
    empty = np.array([])
    if predicted_points.shape[0] <= 5:
        return empty, empty, empty, empty, empty
    
    # Extraer coordenadas
    x_gt, y_gt = ground_truth_points[:, 0], -ground_truth_points[:, 1]
    x_pred, y_pred = predicted_points[:, 0], predicted_points[:, 1]

    # Filtrar prediciones x < 0
    mask = x_pred >= 0.0
    x_pred = x_pred[mask]
    y_pred = y_pred[mask]
    
    if x_pred.shape[0] <= 5:
        return empty, empty, empty, empty, empty
    
    # Filtrar ground truth eliminando puntos con Y mayores que el máximo de la predicción
    max_x_pred = np.max(x_pred)
    mask_gt = x_gt <= max_x_pred
    x_gt, y_gt = x_gt[mask_gt], y_gt[mask_gt]
    
    # Crear interpolador para el ground truth
    t_gt = np.linspace(0, 1, len(x_gt))  # Parámetro normalizado
    interp_x = interp.interp1d(t_gt, x_gt, kind='linear')  # Interpolación lineal
    interp_y = interp.interp1d(t_gt, y_gt, kind='linear')
    
    # Obtener valores interpolados en los mismos t que los predichos
    xmax = np.max(x_gt)
    xmin = np.min(x_gt)
    rango = xmax - xmin
    t_pred = (x_pred.copy() - xmin)/rango
    mask = t_pred > 1.0
    t_pred[mask] = 1.0
    
    x_gt_interp = interp_x(t_pred)
    y_gt_interp = interp_y(t_pred)
    
    # Calcular distancia euclídea
    errors = np.sqrt((x_pred - x_gt_interp) ** 2 + (y_pred - y_gt_interp) ** 2)
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    std_error = np.std(errors)
    
    # Definir rangos de distancia en Y
    ranges = {
        "0-10m": (0, 10),
        "10-20m": (10, 20),
        "20-30m": (20, 30),
        "30-40m": (30, 40),
        "+40m": (40, np.inf)
    }

    # Calcular métricas por rango de X
    errors_by_range = {}
    for label, (x_min, x_max) in ranges.items():
        mask = (x_pred >= x_min) & (x_pred < x_max)
        if np.any(mask):
            errors_in_range = errors[mask]
            errors_by_range[label] = {
                "mean_error": np.mean(errors_in_range),
                "max_error": np.max(errors_in_range),
                "std_error": np.std(errors_in_range)
            }
            
            # Calcular AP para diferentes umbrales
            hit_rates = {}
            for threshold in thresholds:
                hits = errors_in_range < threshold
                hit_rates[f"AP@{threshold}"] = np.mean(hits)
            errors_by_range[label]["hit_rates"] = hit_rates
        else:
            errors_by_range[label] = {
                "mean_error": None,
                "max_error": None,
                "std_error": None,
                "hit_rates": None
            }
            
    plt.clf()
    
    plt.plot(y_gt_interp, x_gt_interp, '-o', label = 'Ground Truth')
    plt.plot(y_pred, x_pred, '.-', label = 'Prediction')
    for i in range(x_pred.shape[0]):
        plt.plot([y_pred[i], y_gt_interp[i]], [x_pred[i], x_gt_interp[i]], '-', color = 'red')

    plt.legend()
    plt.ylim([-10, 50])
    plt.xlim([-10, 10])
    plt.show(block=False)
    plt.pause(0.001)
    
    return errors_by_range, mean_error, max_error, std_error, errors


def get_local_trajectory(img_ts, gt_waypoints):    
    
    # Get closest GPS point
    row_index = np.argmin(np.abs(gt_waypoints[:, 0] - img_ts))
    # Only consider future waypoints
    gt_waypoints = gt_waypoints[row_index:row_index+10*140, :]
    # Get current position
    ts, lat, lon, alt, heading = gt_waypoints[0, :]
    x0, y0, zone_number, zone_letter = utm.from_latlon(lat, lon)
    # Transform GPS points to local coordinates in current frame
    x0_all, y0_all, _, _ = utm.from_latlon(gt_waypoints[:,1], gt_waypoints[:,2], force_zone_number=zone_number, force_zone_letter=zone_letter)
    x0_all -= x0
    y0_all -= y0
    # Adjust depending on heading
    yaw = math.radians(90 - heading)
    yaw_to_adjust = -yaw
    R = np.array([[math.cos(yaw_to_adjust), -math.sin(yaw_to_adjust)], 
                  [math.sin(yaw_to_adjust), math.cos(yaw_to_adjust)]])
    xy_rotated = R @ np.vstack((x0_all, y0_all))
    return xy_rotated[:, ::4]
    #translation = np.array([[1.52377], [0.0]])

    # Apply traslation
    #translatxy_rotated_traslateded_points = xy_rotated + translation
    
    #return translatxy_rotated_traslateded_points[:, ::4]

class rtmaps_python(BaseComponent):

    def __init__(self):
        BaseComponent.__init__(self) 

    def Dynamic(self):
        self.add_input("image_in", rtmaps.types.ANY) 
        
        self.add_property("ground_truth_trajectory", "", subtype=rtmaps.types.FILE)
        self.add_property("model_path", "", subtype=rtmaps.types.FILE)

        self.add_output("prediction", rtmaps.types.IPL_IMAGE)
            
# Birth() will be called once at diagram execution startup
    def Birth(self):
        print("Passing through Birth()")
        # Load Ground truth trajectory
        gt_path = self.get_property("ground_truth_trajectory")
        self.gt_waypoints = np.loadtxt(gt_path, delimiter=';')
        
        # Load model
        model_path = self.get_property("model_path")        
        self.modeloSC = SuperComboRT(model_path, True)   
        
        self.metrics = {
            'errors_by_range' : [], 
            'mean_err' : [], 
            'max_err' : [], 
            'std_err' : [], 
            'errors' : [], 
        }                     
        self.i_dx = 0

# Core() is called every time you have a new inputs available, depending on your chosen reading policy
    def Core(self):
        self.i_dx += 1
        img = self.inputs["image_in"].ioelt.data.image_data # Image in RGB
        img_ts = self.inputs["image_in"].ioelt.ts
        
        # Get trajectory from supercombo
        self.modeloSC.run_model_frame(img)
        rgb_prediction = self.modeloSC.img_plot

        out = rtmaps.types.Ioelt()
        out.data = rtmaps.types.IplImage()
        #out.data.image_data = cv2.cvtColor(bgr_prediction, cv2.COLOR_BGR2RGB)
        out.data.image_data = rgb_prediction
        out.data.color_model = "COLR"  # Color model can also be : RGB, RGBA, YUV, YUVA, GRAY... (SDK C++ for more)
        out.data.channel_seq = "RGB"  # 

        # Get local trayectory
        gt_trajectory = get_local_trajectory(img_ts, self.gt_waypoints)      
        prediction = self.modeloSC.path_preds[-1][0, :, :3]
        
        errors_by_range, mean_err, max_err, std_err, errors = calculate_trajectory_error(prediction, gt_trajectory.T)
        self.metrics['errors_by_range'].append(errors_by_range)
        self.metrics['mean_err'].append(mean_err)
        self.metrics['max_err'].append(max_err)
        self.metrics['std_err'].append(std_err)
        self.metrics['errors'].append(errors.tolist())   
        """if self.i_dx % 20 == 0:
            self.save_json('/home/VICOMTECH/gortiz/Proyectos/Mendi4U/openpilot-pipeline/rtmaps/metrics.json')
        """    
        self.write("prediction", out)
    def save_json(self, path):
        # Guardar en JSON
        with open(path, "w") as f:
            print('Guardando json con métricas')
            json.dump(self.metrics, f, indent=4)
        
# Death() will be called once at diagram execution shutdown
    def Death(self):
        self.video_writer.release()
        
        self.save_json('/home/VICOMTECH/gortiz/Proyectos/Mendi4U/openpilot-pipeline/rtmaps/metrics.json')
    
        print("Passing through Death()")
