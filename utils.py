import json
import numpy as np
import yaml
import os
from pathlib import Path
import scipy.stats as stats



BN_MOMENTUM = 0.1    

def npprint_suppress():
    np.set_printoptions(suppress=True, precision=10)
    
def modified_sigmoid(x, steepness, where_fn_is_05):
    return 1 / (1 + np.exp(-steepness * (x - where_fn_is_05)))

def truncated_normal_sampler(mean, std, lower_bound, upper_bound, size=1):
    if std == 0:
        return np.ones(size) * mean
    a, b = (lower_bound - mean) / std, (upper_bound - mean) / std
    return stats.truncnorm.rvs(a, b, loc=mean, scale=std, size=size)    

def get_subsample_inds(length, subsample_num):
    if subsample_num is None:
        return np.arange(length)
    if subsample_num > length:
        subsample_num = length
    return np.random.permutation(length)[:subsample_num]

class utilitySuite:
    def __init__(self):
        pass
    
    def __new__(self, config=None, create_logger_file=False, config_path=None):
        if config is None:
            if config_path is None:
                self.config = ConfigYAML()
                self.config.kmonitor_enable = 0
            else:
                self.config = ConfigYAML()
                self.config.load_file(config_path)
                if 'kmonitor_enable' not in vars(self.config):
                    self.config.kmonitor_enable = 0
            self.log = Logger('./', '', create_file=create_logger_file)
        else:
            self.config = config
            self.log = Logger(config.save_dir, config.exp_name, 
                              create_file=create_logger_file)
        self.timer = Timer()
        self.dp = DataProcessor()
        self.colorpal = colorPalette()
        self.plt = pltUtils()
        self.kmonitor = keyMonitor(enable=self.config.kmonitor_enable)
        self.rec = ListDict()
        self.logline = self.log.line
        
        return self, self.logline

    
    def mkdir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
class keyMonitor:
    def __init__(self, enable=True) -> None:
        os.environ['KEY_OPTION_ZZR'] = '0'
        os.environ['KEY_OPTION_ONCE_ZZR'] = '0'
        if enable:
            import pynput.keyboard as keyboard
            self.keyboard = keyboard
            
            def show(key):
                if "{0}".format(key) == "Key.alt_r":
                    if os.environ['KEY_OPTION_ZZR'] == '0':
                        os.environ['KEY_OPTION_ZZR'] = '1'
                    else:
                        os.environ['KEY_OPTION_ZZR'] = '0'
                    print('[keyMonitor]:' + os.environ['KEY_OPTION_ZZR'])
                elif "{0}".format(key) in list(map(lambda x: "'"+str(x)+"'", range(10))):
                    os.environ['KEY_OPTION_ZZR'] = "{0}".format(key)[1]
                    print('[keyMonitor]:' + os.environ['KEY_OPTION_ZZR'])
                elif "{0}".format(key) == "Key.up":
                    os.environ['KEY_OPTION_ONCE_ZZR'] = '1'
                elif "{0}".format(key) == "Key.down":
                    os.environ['KEY_OPTION_ONCE_ZZR'] = '2'
                        
            listener = keyboard.Listener(on_press = show)    
            listener.start()
        
    def option(self):
        return os.environ['KEY_OPTION_ZZR']
    
    def option_once(self):
        option = os.environ['KEY_OPTION_ONCE_ZZR']
        os.environ['KEY_OPTION_ONCE_ZZR'] = '0'
        return option
    
class pltUtils:
    def __init__(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        self.plt = plt
        self.GridSpec = GridSpec
        self.win_closed = False
        
    def get_fig(self, grid=[1, 1], figsize=[8, 6], dpi=100, gridline=False):
        fig = self.plt.figure(figsize=figsize, dpi=dpi)
        self.fig = fig
        self.fig.tight_layout()
        self.gs = self.GridSpec(grid[0], grid[1])
        self.axs = []
        for ind in range(np.prod(grid)):
            self.axs.append(fig.add_subplot(self.gs[ind]))
        if gridline:
            for ind in range(len(self.axs)):
                self.grid(ind)
            
        def handle_close(evt):
            self.win_closed = True
        self.fig.canvas.mpl_connect('close_event', handle_close)
        return self.axs
    
    def equal(self, ax_num):
        self.axs[ax_num].axis('equal')
        
    def background(self, color, ax_num):
        self.axs[ax_num].set_facecolor(color)
    
    def box(self, ax_num):
        self.axs[ax_num].set_aspect('equal', 'box')
    
    def grid(self, ax_num):
        self.axs[ax_num].grid(which='both', axis='both')
    
    def hide_xy(self, ax_num):
        self.axs[ax_num].get_xaxis().set_visible(False)
        self.axs[ax_num].get_yaxis().set_visible(False)

    def colorbar(self, ax_num, cmap='viridis', data_ind=0, location='right', label=''):
        if len(self.axs[ax_num].collections) > 0:
            cbar = self.plt.colorbar(self.axs[ax_num].collections[data_ind], 
                                     ax=self.axs[ax_num], cmap=cmap, 
                                     location=location, label=label)
        elif len(self.axs[ax_num].images) > 0:
            cbar = self.plt.colorbar(self.axs[ax_num].images[data_ind], 
                                     ax=self.axs[ax_num], cmap=cmap, 
                                     location=location, label=label)
        
            
    def y(self, ax_num, limit):
        self.axs[ax_num].set_ylim(limit)
    
    def x(self, ax_num, limit):
        self.axs[ax_num].set_xlim(limit)
        
    def show(self):
        self.plt.show()
    
    def show_pause(self):
        self.plt.draw()
        while self.plt.waitforbuttonpress(0.2) is None:
            if self.win_closed:
                break
        self.plt.close(self.fig)
        self.win_closed = False
    
    def save_fig(self, filename):
        self.plt.savefig(filename, bbox_inches='tight')
        
    def close_all(self):
        self.plt.close('all')
        
        
class ListDict:
    def __init__(self) -> None:
        pass

    def init(self, *keys):
        for key in keys:
            setattr(self, key, []) 
    
    def get_keys(self):
        return list(vars(self).keys())
    
    def list(self):
        print(self.get_keys())
            
    def pop(self, *keys, index=0):
        if len(keys) == 0:
            keys = self.get_keys()
        for key in keys:
            # print(key)
            getattr(self, key).pop(index)
    
    def save(self, *keys, save_dir=''):
        for key in keys:
            np.savez(save_dir + key, *getattr(self, key))
            
    def load(self, *keys, save_dir=''):
        for key in keys:
            setattr(self, key, list(np.load(save_dir + key + '.npz', allow_pickle=True).values()))
    
    def load_onefile_old(self, save_dir='', filename = 'data_record'):
        d = np.load(save_dir + filename + '.npz', allow_pickle=True)['arr_0'][()]
        for key in list(d.keys()):
            if hasattr(d[key], "__len__"):
                setattr(self, key, list(d[key]))
            else:
                setattr(self, key, d[key])
                
    def save_onefile(self, *keys, save_dir='', filename = 'data_record', compress=False):
        if len(keys) == 0:
            keys = self.get_keys()
        d = {}
        for key in keys:
            d[key] = {key: getattr(self, key)}
        if compress:
            np.savez_compressed(save_dir + filename, **d)
        else:
            np.savez(save_dir + filename, **d)
            
    def load_onefile(self, *keys, save_dir='', filename = 'data_record'):
        d = np.load(save_dir + filename + '.npz', allow_pickle=True)
        if len(keys) == 0:
            keys = list(d.keys())
        for key in keys:
            if hasattr(d[key][()][key], "__len__"):
                setattr(self, key, list(d[key][()][key]))
            else:
                setattr(self, key, d[key][()][key])
    
class Timer:
    def __init__(self, enable=True) -> None:
        self.enable = enable
        self.times = {}
        import time
        self.time = time
        
    def tic(self, time_name=None):
        if self.enable:
            if time_name is None:
                time_name = str(len(list(self.times.keys())))
            self.times[time_name] = self.time.time()
            
    def toc(self, name='', time_name=None, Hz=False, show=True):
        if self.enable:
            if time_name is None:
                time_name = str(len(list(self.times.keys())) - 1)
            if Hz: 
                ret = 1/(self.time.time() - self.times[time_name])
            else:
                ret = self.time.time() - self.times[time_name]
            if Hz and show: print(name, ret, 'Hz')
            elif show: print(name, ret, 's')
            return ret
    
    def toctic(self, name='', time_name=None, Hz=False, show=True, time_name2=None):
        ret = self.toc(name, time_name, Hz, show)
        self.tic(time_name2)
        return ret
    
    def ding(self):
        if self.enable:
            print()


class colorPalette:
    def __init__(self, colorset=None) -> None:
        self.colorset = colorset
        if colorset is None:
            self.colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', 
                           '#ffff33', '#a65628', '#f781bf', '#999999', '#000000',]
        else:
            import seaborn as sns
            self.sns = sns
            self.colors = self.sns.color_palette(self.colorset).as_hex()
        
    def hex2rgb(self, hex):
        return [int(hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]
    
    def hex2rgb_normalize(self, hex):
        return [int(hex.lstrip('#')[i:i+2], 16)/255.0 for i in (0, 2, 4)]
        
    def rgb(self, color_ind):
        if isinstance(color_ind, str):
            if self.colorset == "Set1" or self.colorset == None:
                color_ind = ['r', 'b', 'g', 'p', 'o', 'y', 'br', 'pi', 'w'].index(color_ind)
        return self.hex2rgb(self.colors[color_ind])
    
    def rgb_normalize(self, color_ind):
        if isinstance(color_ind, str):
            if self.colorset == "Set1" or self.colorset == None:
                color_ind = ['r', 'b', 'g', 'p', 'o', 'y', 'br', 'pi', 'w'].index(color_ind)
        return self.hex2rgb_normalize(self.colors[color_ind])
    
    
    
class Logger:
    def __init__(self, save_dir, experiment_name, create_file=True) -> None:
        from io import StringIO
        self.s = StringIO()
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        if create_file:
            self.create_file(experiment_name)
    
    def create_file(self, experiment_name):
        import datetime
        print(self.experiment_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(self.save_dir + experiment_name + '.txt', "a") as tgt:
            tgt.writelines('\n' + '-' * 80 + '\n')
            tgt.writelines(experiment_name + ' ' + str(datetime.datetime.now()) + '\n')
            
    def write_file(self, file):
        with open(file, "r") as src:
            with open(self.save_dir + self.experiment_name + '.py', "w") as tgt:
                tgt.write(src.read())
    
    def line(self, *line, print_line=True):
        if print_line: print(*line)
        print(*line, file = self.s)
        with open(self.save_dir + self.experiment_name + '.txt', "a") as tgt:
            tgt.writelines(self.s.getvalue())
        self.s.truncate(0)
        self.s.seek(0)
            

class open3dUtils:
    def __init__(self):
        from scipy.spatial.transform import Rotation
        import open3d as o3d
        self.Rotation = Rotation
        self.o3d = o3d
        self.object_list = []
        self.show_axis = True

    def create_camera_poses(self, extrinsic, size=1, color=[1, 0, 0], aspect_ratio=0.3, alpha=0.15, linewidths=0.1):
        focal_len_scaled = size
        vertex_std = np.array([[0, 0, 0, 1],
                                [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                                [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                                [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                                [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                                [0, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                                [0, 0, focal_len_scaled, 1],])
        T = np.eye(4)
        T[:3, :3] = self.Rotation.from_euler('Y', [180], degrees=True).as_matrix()
        for ind in range(vertex_std.shape[0]):
            vertex_std[ind] = T @ vertex_std[ind]
        vertex_transformed = vertex_std @ extrinsic.T
        
        points = vertex_transformed[:, :3]
        lines = [[0, 1], [0, 2], [1, 2], [2, 3], [0, 3], [0, 4], [3, 4], [1, 4], [5, 6], [0, 6]]
        colors = [color for i in range(len(lines))]
        line_set = self.o3d.geometry.LineSet()
        line_set.points = self.o3d.utility.Vector3dVector(points)
        line_set.lines = self.o3d.utility.Vector2iVector(lines)
        line_set.colors = self.o3d.utility.Vector3dVector(colors)
        return line_set
    
    def add_object(self, object):
        self.object_list.append(object)
    
    def clear_object(self):
        self.object_list = []
        
    def show(self, background_color=np.asarray([0.0, 0.0, 0.0])):
        viewer = self.o3d.visualization.Visualizer()
        viewer.create_window()
        for geometry in self.object_list:
            viewer.add_geometry(geometry)
        opt = viewer.get_render_option()
        opt.show_coordinate_frame = self.show_axis
        opt.background_color = background_color
        viewer.run()
        viewer.destroy_window()
        del viewer
        del opt




def readTXT(filename):
    with open(filename) as f:
        lines = f.readlines()
    return lines

class ConfigJSON():
    def __init__(self) -> None:
        self.d = {}
    
    def load_file(self, filename):
        with open(filename, 'r') as f:
            self.d = json.load(f)
    
    def save_file(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.d, f, ensure_ascii=False, indent=4)

class ConfigYAML():
    """
    Config class for yaml file
    Able to load and save yaml file to and from python object
    """
    def __init__(self) -> None:
        pass
    
    def load_file(self, filename):
        d = yaml.safe_load(Path(filename).read_text())
        for key in d: 
            setattr(self, key, d[key]) 
    
    def save_file(self, filename):
        d = vars(self)
        class_d = vars(self.__class__)
        d_out = {}
        for key in list(class_d.keys()):
            if not (key.startswith('__') or \
                    key.startswith('load_file') or \
                    key.startswith('save_file')):
                if isinstance(class_d[key], np.ndarray):
                    d_out[key] = class_d[key].tolist()
                else:
                    d_out[key] = class_d[key]
        for key in list(d.keys()):
            if not (key.startswith('__') or \
                    key.startswith('load_file') or \
                    key.startswith('save_file')):
                if isinstance(d[key], np.ndarray):
                    d_out[key] = d[key].tolist()
                else:
                    d_out[key] = d[key]
        with open(filename, 'w+') as ff:
            yaml.dump_all([d_out], ff)
            
            
class DataProcessor():
    def __init__(self) -> None:
        pass
    
    def find_range(self, data):
        if len(data.shape) == 1:
            return np.array([np.min(data), np.max(data)])
        range_min = []
        range_max = []
        for k in range(data.shape[1]):
            range_min.append(np.min(data[:, k]))
            range_max.append(np.max(data[:, k]))
        return np.array([range_min, range_max])

    def find_larger_normal_params(self, range1, range2):
        range_ret = range1.copy()
        for k in range(range_ret.shape[0]):
            range_ret[k, 0] = np.max([range1[k, 0], 
                                      range2[k, 0], 
                                      range1[k, 0]+range1[k, 1]-range2[k, 1],
                                      range2[k, 0]+range2[k, 1]-range1[k, 1]])
            range_ret[k, 1] = np.min([range1[k, 1], range2[k, 1]])
        return range_ret
        # for k in range(range_ret.shape[1]):
        #     range_ret[0, k] = np.min([range1[0, k], range2[0, k]])
        #     range_ret[1, k] = np.max([range1[1, k], range2[1, k]])
        # return range_ret
    
    def two_pi_warp(self, angles):
        twp_pi = 2 * np.pi
        return float((angles + twp_pi) % (twp_pi))
    
    def data_normalize(self, data):
        data_min = np.min(data)
        data = data - data_min
        data_max = np.max(data)
        data = data / data_max
        return data, [float(data_max), float(data_min)]
    
    def runtime_normalize(self, data, params):
        return (data - params[1]) / params[0]
    
    def de_normalize(self, data, params):
        return data * params[0] + params[1]

class DrivableCritic():
    def __init__(self, yaml_dir, yaml_filename):
        import cv2
        with open(yaml_dir + yaml_filename) as f:
            map_yaml = yaml.load(f, Loader=yaml.FullLoader)
        # self.img = np.load(map_yaml['image'] + '.npy')
        self.img = cv2.imread(yaml_dir + map_yaml['image'], cv2.IMREAD_GRAYSCALE)
        self.img_ori = np.array(map_yaml['origin'])
        self.img_res = np.array(map_yaml['resolution'])
        self.x_in_m = self.img.shape[1] * self.img_res
        self.y_in_m = self.img.shape[0] * self.img_res
        
    def get_normalize_params(self):
        params = np.zeros((4, 2))
        params[0, 1] = self.img_ori[0]
        params[0, 0] = self.x_in_m
        params[1, 1] = self.img_ori[1]
        params[1, 0] = self.y_in_m
        params[2, 0] = np.pi * 2
        params[3, 0] = 30
        return params
        
    def pose_2_colrow(self, pose):
        colrow = (pose[:, :2] - self.img_ori[:2]) / self.img_res
        return np.int32(colrow)

    def normalized_pose_2_rowcol(self, normalized_pose):
        return np.int32([(1 - normalized_pose[:, 1]) * self.img.shape[0],
                  normalized_pose[:, 0] * self.img.shape[1]]).transpose(1, 0)

    def normalized_pose_find_drivable(self, normalized_pose):
        rowcol = self.normalized_pose_2_rowcol(normalized_pose)
        return self.img[rowcol[:, 0], rowcol[:, 1]]
    
    def normalized_pose_2_xy(self, normalized_pose):
        img_size_xy = np.array([self.img.shape[1], self.img.shape[0]])
        return (normalized_pose * img_size_xy) * self.img_res + self.img_ori[:2]


class plotlyUtils:
    def __init__(self, renderer = None) -> None:
        import plotly.graph_objects as go
        import plotly.io as pio
        self.pio = pio
        self.go = go
        if renderer is not None:
            self.pio.renderers.default = renderer
        
    def _extrinsic2pyramid_mesh(self, extrinsic, color='r', focal_len_scaled=1, aspect_ratio=1.3, alpha=0.15, linewidths=0.1):
        vertex_std = np.array([[0, 0, 0, 1],
                                [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                                [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                                [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                                [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        return vertex_transformed

    def _create_camera_pose(self, vertex, color=[1, 0, 0]):
        return self.go.Mesh3d(
            # 8 vertices of a cube
            x=vertex[:, 0],
            y=vertex[:, 1],
            z=vertex[:, 2],

            # i, j and k give the vertices of triangles
            i = [0, 0, 0, 0],
            j = [1, 2, 3, 4],
            k = [2, 3, 4, 1],
            showscale=True,
            opacity=0.5,
            facecolor=[list(color)] * 5
        )
        
    def create_point(self, xyz, color=[1, 0, 0]):
        return self.go.Scatter3d(
            x = xyz[0],
            y = xyz[1],
            z = xyz[2],
            opacity = 1,
            surfacecolor = color)

    def add_plot_data(self, data_list, data_in, color):
        for ind in range(len(data_in)):
            vertex = self._extrinsic2pyramid_mesh(data_in[ind], aspect_ratio=0.5)
            data_list.append(self._create_camera_pose(vertex, color=color))
        return data_list
    
    def plot3d_show(self, data_list, axis_limits):
        fig = self.go.Figure(data=data_list)
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            scene = dict(
                xaxis = dict(nticks=4, range=[axis_limits[0], axis_limits[1]],),
                yaxis = dict(nticks=4, range=[axis_limits[2], axis_limits[3]],),
                zaxis = dict(nticks=4, range=[axis_limits[4], axis_limits[5]],),
                aspectmode='data'),
            width = 1350
            )
        fig.show()