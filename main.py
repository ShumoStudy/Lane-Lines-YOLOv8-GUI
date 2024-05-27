from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, ops
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from ultralytics.yolo.utils.torch_utils import smart_inference_mode
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.checks import check_imshow
from ultralytics.yolo.cfg import get_cfg
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtCore import QTimer, QThread, Signal, QObject, QPoint, Qt
from ui.CustomMessageBox import MessageBox
from ui.home import Ui_MainWindow
from UIFunctions import *
from collections import defaultdict
from pathlib import Path
import numpy as np
import time
import json
import torch
import sys
import cv2
import os


class YoloPredictor(BasePredictor, QObject):
    yolo2main_pre_img = Signal(np.ndarray)       # 原始图像信号
    yolo2main_res_img = Signal(np.ndarray)       # 测试结果信号
    yolo2main_status_msg = Signal(str)           # 检测/暂停/停止/测试完成/错误报告信号
    yolo2main_fps = Signal(str)                  # 帧率信号
    yolo2main_labels = Signal(dict)              # 检测到的目标结果（每个类别的数量）
    yolo2main_progress = Signal(int)             # 完成度信号
    yolo2main_class_num = Signal(int)            # 检测到的类别数量信号
    yolo2main_target_num = Signal(int)           # 检测到的目标数量信号

    def __init__(self, cfg=DEFAULT_CFG, overrides=None): 
        super(YoloPredictor, self).__init__()       # 调用父类的构造函数，确保 BasePredictor 的初始化。
        QObject.__init__(self)                      # 调用 QObject 的构造函数，确保其属性和方法可用。

        self.args = get_cfg(cfg, overrides)         # 获取配置参数：self.args 使用 get_cfg 函数获取配置参数，
        # 传递默认配置 cfg 和可选的覆盖配置 overrides。
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task      # project 确定项目路径，
        # 使用 self.args.project 或者 SETTINGS['runs_dir'] 和任务名组合成路径。
        name = f'{self.args.mode}'                  # name 设置为模式名称。
        self.save_dir = increment_path(Path(project) / name, exist_ok=self.args.exist_ok)
        # self.save_dir 通过 increment_path 函数生成保存目录路径，确保路径唯一性，如果 self.args.exist_ok 为 True，则允许覆盖现有路径。
        self.done_warmup = False        #初始化暖机状态：self.done_warmup 设置为 False，表示未完成暖机。
        if self.args.show:
            self.args.show = check_imshow(warn=True)
        # 检查显示选项：如果 self.args.show 为 True，调用 check_imshow 函数检查是否可以显示图像，并设置 self.args.show 为检查结果。

        # GUI 参数
        self.used_model_name = None     # 要使用的检测模型名称
        self.new_model_name = None      # 实时更改的模型
        self.source = ''                # 输入源
        self.stop_dtc = False           # 终止检测
        self.continue_dtc = True        # 暂停
        self.save_res = False           # 保存测试结果
        self.save_txt = False           # 保存标签(txt)文件
        self.iou_thres = 0.45           # iou 阈值
        self.conf_thres = 0.25          # 置信度阈值
        self.speed_thres = 10           # 延迟，毫秒
        self.labels_dict = {}           # 返回结果字典
        self.progress_value = 0         # 进度条值

        # 属性初始化
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None    # 视频路径和视频写入器
        self.annotator = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.callbacks = defaultdict(list, callbacks.default_callbacks)  # add callbacks
        # 回调函数：self.callbacks 初始化为一个 defaultdict，默认值为 list，
        # 并且使用 callbacks.default_callbacks 进行初始化。回调函数用于在某些事件发生时执行特定的操作。
        callbacks.add_integration_callbacks(self)
        # 添加集成回调：调用 callbacks.add_integration_callbacks(self) 方法，
        # 为 self（即当前的 YoloPredictor 实例）添加一些默认的集成回调函数。
        # 这些回调函数可以在特定的事件（如模型推理、数据处理等）发生时被调用。

    # main for detect
    @smart_inference_mode()
    def run(self):
        try:
            if self.args.verbose:
                LOGGER.info('')

            # 设置模型：发送加载模型状态消息。如果尚未加载模型，则调用 setup_model （YOLOV8）方法加载新模型，并更新 used_model_name。
            self.yolo2main_status_msg.emit('Loding Model...')
            if not self.model:
                self.setup_model(self.new_model_name)
                self.used_model_name = self.new_model_name

            # 设置输入源：调用 setup_source 方法设置数据源
            self.setup_source(self.source if self.source is not None else self.args.source)

            # 检查保存路径和标签：如果需要保存结果或标签文件，则创建相应目录。
            if self.save_res or self.save_txt:
                (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # 模型预热：如果尚未预热模型，调用 warmup 方法进行预热，并设置 done_warmup 为 True。
            if not self.done_warmup:
                self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                self.done_warmup = True

            self.seen, self.windows, self.dt, self.batch = 0, [], (ops.Profile(), ops.Profile(), ops.Profile()), None
            # 初始化：初始化一些变量，包括已处理的帧数、窗口列表、计时器（用于记录不同阶段的耗时），以及批处理变量。

            # 准备开始目标检测
            # for batch in self.dataset:

            # 初始化计数和时间：初始化帧计数器 count 和起始时间 start_time，并将数据集转换为迭代器。
            count = 0                       # run location frame
            start_time = time.time()        # used to calculate the frame rate
            batch = iter(self.dataset)
            while True:
                # 终止检测：检查是否需要终止检测，如果需要，释放视频写入器并发送终止消息。
                if self.stop_dtc:
                    if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                        self.vid_writer[-1].release()  # release final video writer
                    self.yolo2main_status_msg.emit('Detection terminated!')
                    break
                
                # 中途更改模型：如果检测到模型更改，则重新设置模型并更新 used_model_name。
                if self.used_model_name != self.new_model_name:  
                    # self.yolo2main_status_msg.emit('Change Model...')
                    self.setup_model(self.new_model_name)
                    self.used_model_name = self.new_model_name
                
                # 暂停开关：检查是否暂停，如果没有暂停，发送检测中消息，并获取下一批数据。
                if self.continue_dtc:
                    # time.sleep(0.001)
                    self.yolo2main_status_msg.emit('Detecting...')
                    batch = next(self.dataset)  # next data

                    self.batch = batch
                    path, im, im0s, vid_cap, s = batch
                    visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.args.visualize else False

                    # 批处理数据：处理批数据，包括路径、图像、原始图像、视频捕获对象等。根据需要设置可视化路径。
                    count += 1              # frame count +1
                    if vid_cap:
                        all_count = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)   # 获取视频文件总帧数
                    else:
                        all_count = 1
                    self.progress_value = int(count/all_count*1000)         # progress bar(0~1000) 计算处理进度并更新进度条
                    if count % 5 == 0 and count >= 5:                     # 没5帧计算并发送帧率信息
                        self.yolo2main_fps.emit(str(int(5/(time.time()-start_time))))
                        start_time = time.time()                        # 更新开始处理时间
                    
                    # preprocess 预处理
                    with self.dt[0]:
                        im = self.preprocess(im)
                        if len(im.shape) == 3:
                            im = im[None]  # expand for batch dim
                    # inference 推理
                    with self.dt[1]:
                        preds = self.model(im, augment=self.args.augment, visualize=visualize)
                    # postprocess 后处理
                    with self.dt[2]:
                        self.results = self.postprocess(preds, im, im0s)
                    # 按顺序执行并计时
                    # 可视化、保存和写入结果：循环处理每张图像，计算预处理、推理和后处理的速度，并获取路径和原始图像。
                    n = len(im)     # To be improved: support multiple img
                    for i in range(n):
                        self.results[i].speed = {
                            'preprocess': self.dt[0].dt * 1E3 / n,
                            'inference': self.dt[1].dt * 1E3 / n,
                            'postprocess': self.dt[2].dt * 1E3 / n}
                        p, im0 = (path[i], im0s[i].copy()) if self.source_type.webcam or self.source_type.from_img \
                            else (path, im0s.copy())
                        p = Path(p)     # the source dir

                        # s:::   video 1/1 (6/6557) 'path':
                        # must, to get boxs\labels
                        label_str = self.write_results(i, self.results, (p, im, im0))   # labels   /// original :s += 
                        
                        # labels and nums dict
                        class_nums = 0
                        target_nums = 0
                        self.labels_dict = {}
                        if 'no detections' in label_str:
                            pass
                        else:
                            for ii in label_str.split(',')[:-1]:
                                nums, label_name = ii.split('~')
                                self.labels_dict[label_name] = int(nums)
                                target_nums += int(nums)
                                class_nums += 1
                        # 标签和数量字典：写入结果并解析标签字符串，统计每个类别的目标数量，并更新标签字典

                        # 保存和发送结果：根据需要保存图像或视频结果，并发送检测后的图像、检测前的图像、类别数量和目标数量。
                        # 如果设置了速度阈值，则延迟相应时间。
                        # save img or video result
                        if self.save_res:
                            self.save_preds(vid_cap, i, str(self.save_dir / p.name))

                        # 检测完成：检查是否已处理所有帧，如果是，释放视频写入器并发送检测完成消息。
                        self.yolo2main_res_img.emit(im0) # after detection
                        self.yolo2main_pre_img.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])   # Before testing
                        # self.yolo2main_labels.emit(self.labels_dict)        # webcam need to change the def write_results
                        self.yolo2main_class_num.emit(class_nums)
                        self.yolo2main_target_num.emit(target_nums)

                        if self.speed_thres != 0:
                            time.sleep(self.speed_thres/1000)   # delay , ms

                    self.yolo2main_progress.emit(self.progress_value)   # progress bar

                # Detection completed
                if count + 1 >= all_count:
                    if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                        self.vid_writer[-1].release()  # 释放视频写入
                    self.yolo2main_status_msg.emit('Detection completed')
                    break

        except Exception as e:
            pass
            print(e)
            self.yolo2main_status_msg.emit('%s' % e)
        # 捕捉异常，但是 发生异常会忽略异常，通过信号把异常信息发送出去


    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))
    # 开始绘制 line_width 设置了线条的厚度 example 来自模型的注释类别的名称

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        # 这个方法将图像从 NumPy 数组转换为 PyTorch 张量，并将其移动到模型的设备上。
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        # 如果模型使用 fp16（半精度浮点数），则将图像转换为 fp16，否则转换为浮点数（fp32）。
        img /= 255  # 将像素值从 0-255 归一化到 0.0-1.0 的范围。
        return img


    def postprocess(self, preds, img, orig_img):
        ### important 对预测结果进行后处理
        # 首先使用非极大值抑制（Non-Max Suppression）来过滤重叠的检测框。
        preds = ops.non_max_suppression(preds,
                                        self.conf_thres,
                                        self.iou_thres,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img          # 如果 orig_img 是列表，则取出对应的原始图像。
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            path, _, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path      # 从批处理路径中获取图像路径。
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
            # 创建一个 Results 对象，并将其添加到结果列表中。
        # print(results)
        return results

    def write_results(self, idx, results, batch):
        p, im, im0 = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        imc = im0.copy() if self.args.save_crop else im0
        if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1         # attention
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        # log_string += '%gx%g ' % im.shape[2:]         # !!! don't add img size~
        self.annotator = self.get_annotator(im0)

        det = results[idx].boxes  # TODO: make boxes inherit from tensors

        if len(det) == 0:
            return f'{log_string}(no detections), ' # if no, send this~~

        for c in det.cls.unique():
            n = (det.cls == c).sum()  # detections per class
            log_string += f"{n}~{self.model.names[int(c)]},"   #   {'s' * (n > 1)}, "   # don't add 's'
        # now log_string is the classes 👆


        # write
        for d in reversed(det):
            cls, conf = d.cls.squeeze(), d.conf.squeeze()
            # 如果 self.save_txt 为真，则将检测结果写入到 self.txt_path.txt 文件中。写入的内容包括类别、边界框坐标（如果有需要，还包括置信度）。
            if self.save_txt:  # Write to file
                line = (cls, *(d.xywhn.view(-1).tolist()), conf) \
                    if self.args.save_conf else (cls, *(d.xywhn.view(-1).tolist()))  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
            # 如果需要保存结果图像（由 self.save_res、self.args.save_crop、self.args.show 或 True 控制），
            # 则在图像上绘制边界框，并可能添加其他注释（如类别名称和ID）。但请注意，此部分的代码在提供的代码片段中被截断了。
            if self.save_res or self.args.save_crop or self.args.show or True:  # Add bbox to image(must)
                c = int(cls)  # integer class
                name = f'id:{int(d.id.item())} {self.model.names[c]}' if d.id is not None else self.model.names[c]
                label = None if self.args.hide_labels else (name if self.args.hide_conf else f'{name} {conf:.2f}')
                self.annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))
            if self.args.save_crop:
                save_one_box(d.xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string
        


class MainWindow(QMainWindow, Ui_MainWindow):
    main2yolo_begin_sgl = Signal()  # The main window sends an execution signal to the yolo instance
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # basic interface
        self.setupUi(self)
        self.setAttribute(Qt.WA_TranslucentBackground)  # rounded transparent
        self.setWindowFlags(Qt.FramelessWindowHint)  # Set window flag: hide window borders
        UIFuncitons.uiDefinitions(self)
        # Show module shadows
        UIFuncitons.shadow_style(self, self.Class_QF, QColor(162,129,247))
        UIFuncitons.shadow_style(self, self.Target_QF, QColor(251, 157, 139))
        UIFuncitons.shadow_style(self, self.Fps_QF, QColor(170, 128, 213))
        UIFuncitons.shadow_style(self, self.Model_QF, QColor(64, 186, 193))
        


        # read model folder
        self.pt_list = os.listdir('./models')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./models/' + x))   # sort by file size
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        self.Qtimer_ModelBox = QTimer(self)     # Timer: Monitor model file changes every 2 seconds
        self.Qtimer_ModelBox.timeout.connect(self.ModelBoxRefre)
        self.Qtimer_ModelBox.start(2000)

        # Yolo-v8 thread
        self.yolo_predict = YoloPredictor()                           # Create a Yolo instance
        self.select_model = self.model_box.currentText()                   # default model
        self.yolo_predict.new_model_name = "./models/%s" % self.select_model  
        self.yolo_thread = QThread()                                  # Create yolo thread
        self.yolo_predict.yolo2main_pre_img.connect(lambda x: self.show_image(x, self.pre_video)) 
        self.yolo_predict.yolo2main_res_img.connect(lambda x: self.show_image(x, self.res_video))
        self.yolo_predict.yolo2main_status_msg.connect(lambda x: self.show_status(x))             
        self.yolo_predict.yolo2main_fps.connect(lambda x: self.fps_label.setText(x))              
        # self.yolo_predict.yolo2main_labels.connect(self.show_labels)                            
        self.yolo_predict.yolo2main_class_num.connect(lambda x:self.Class_num.setText(str(x)))         
        self.yolo_predict.yolo2main_target_num.connect(lambda x:self.Target_num.setText(str(x)))       
        self.yolo_predict.yolo2main_progress.connect(lambda x: self.progress_bar.setValue(x))     
        self.main2yolo_begin_sgl.connect(self.yolo_predict.run)     
        self.yolo_predict.moveToThread(self.yolo_thread)              

        # Model parameters
        self.model_box.currentTextChanged.connect(self.change_model)     
        self.iou_spinbox.valueChanged.connect(lambda x:self.change_val(x, 'iou_spinbox'))    # iou box
        self.iou_slider.valueChanged.connect(lambda x:self.change_val(x, 'iou_slider'))      # iou scroll bar
        self.conf_spinbox.valueChanged.connect(lambda x:self.change_val(x, 'conf_spinbox'))  # conf box
        self.conf_slider.valueChanged.connect(lambda x:self.change_val(x, 'conf_slider'))    # conf scroll bar
        self.speed_spinbox.valueChanged.connect(lambda x:self.change_val(x, 'speed_spinbox'))# speed box
        self.speed_slider.valueChanged.connect(lambda x:self.change_val(x, 'speed_slider    '))  # speed scroll bar

        # Prompt window initialization
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')
        self.Model_name.setText(self.select_model)
        
        # 选择检测源
        self.src_file_button.clicked.connect(self.open_src_file)  # select local file
        # self.src_cam_button.clicked.connect(self.show_status("The function has not yet been implemented."))#chose_cam
        # self.src_rtsp_button.clicked.connect(self.show_status("The function has not yet been implemented."))#chose_rtsp

        # 开始测试按钮
        self.run_button.clicked.connect(self.run_or_continue)   # pause/start
        self.stop_button.clicked.connect(self.stop)             # termination

        # 其他函数按钮
        self.save_res_button.toggled.connect(self.is_save_res)  # save image option
        self.save_txt_button.toggled.connect(self.is_save_txt)  # Save label option
        self.ToggleBotton.clicked.connect(lambda: UIFuncitons.toggleMenu(self, True))   # left navigation button
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True))   # top right settings button
        
        # initialization
        self.load_config()

    # The main window displays the original image and detection results
    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep the original data ratio
            if iw/w > ih/h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            # 颜色空间转换
            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            # 把opencv图像转换成QT图像
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            # 再标签上显示图像
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    # Control start/pause
    def run_or_continue(self):
        if self.yolo_predict.source == '':
            self.show_status('Please select a video source before starting detection...')
            self.run_button.setChecked(False)
        else:
            self.yolo_predict.stop_dtc = False
            if self.run_button.isChecked():
                self.run_button.setChecked(True)    # start button
                self.save_txt_button.setEnabled(False)  # It is forbidden to check and save after starting the detection
                self.save_res_button.setEnabled(False)
                self.show_status('Detecting...')           
                self.yolo_predict.continue_dtc = True   # Control whether Yolo is paused
                if not self.yolo_thread.isRunning():
                    self.yolo_thread.start()
                    self.main2yolo_begin_sgl.emit()

            else:
                self.yolo_predict.continue_dtc = False
                self.show_status("Pause...")
                self.run_button.setChecked(False)    # start button

    # bottom status bar information
    def show_status(self, msg):
        self.status_bar.setText(msg)
        if msg == 'Detection completed' or msg == '检测完成':
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)    
            self.progress_bar.setValue(0)
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()         # end process
        elif msg == 'Detection terminated!' or msg == '检测终止':
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)    
            self.progress_bar.setValue(0)
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()         # end process
            self.pre_video.clear()           # clear image display  
            self.res_video.clear()          
            self.Class_num.setText('--')
            self.Target_num.setText('--')
            self.fps_label.setText('--')

    # select local file
    def open_src_file(self):
        config_file = 'config/fold.json'    
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']     
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv *.jpg *.png)")
        if name:
            self.yolo_predict.source = name
            self.show_status('Load File：{}'.format(os.path.basename(name))) 
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)  
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.stop()


    def is_save_res(self):
        if self.save_res_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status('NOTE: Run image results are not saved.')
            self.yolo_predict.save_res = False
        elif self.save_res_button.checkState() == Qt.CheckState.Checked:
            self.show_status('NOTE: Run image results will be saved.')
            self.yolo_predict.save_res = True
    
    # Save test result button -- label (txt)
    def is_save_txt(self):
        if self.save_txt_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status('NOTE: Labels results are not saved.')
            self.yolo_predict.save_txt = False
        elif self.save_txt_button.checkState() == Qt.CheckState.Checked:
            self.show_status('NOTE: Labels results will be saved.')
            self.yolo_predict.save_txt = True

    # Configuration initialization  ~~~wait to change~~~
    def load_config(self):
        # 初始化 检测文件是否存在 如果不存在 就 重新创建
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33   
            rate = 10
            save_res = 0   
            save_txt = 0    
            new_config = {"iou": iou,
                          "conf": conf,
                          "rate": rate,
                          "save_res": save_res,
                          "save_txt": save_txt
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 5:
                iou = 0.26
                conf = 0.33
                rate = 10
                save_res = 0
                save_txt = 0
            else:
                iou = config['iou']
                conf = config['conf']
                rate = config['rate']
                save_res = config['save_res']
                save_txt = config['save_txt']
        self.save_res_button.setCheckState(Qt.CheckState(save_res))
        self.yolo_predict.save_res = (False if save_res==0 else True )
        self.save_txt_button.setCheckState(Qt.CheckState(save_txt)) 
        self.yolo_predict.save_txt = (False if save_txt==0 else True )
        self.run_button.setChecked(False)  
        self.show_status("Welcome~")

    # Terminate button and associated state
    def stop(self):
        if self.yolo_thread.isRunning():
            self.yolo_thread.quit()         # end thread
        self.yolo_predict.stop_dtc = True
        self.run_button.setChecked(False)    # start key recovery
        self.save_res_button.setEnabled(True)   # Ability to use the save button
        self.save_txt_button.setEnabled(True)   # Ability to use the save button
        self.pre_video.clear()           # clear image display
        self.res_video.clear()           # clear image display
        self.progress_bar.setValue(0)
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')

    # Change detection parameters
    def change_val(self, x, flag):
        if flag == 'iou_spinbox':
            self.iou_slider.setValue(int(x*100))    # The box value changes, changing the slider
        elif flag == 'iou_slider':
            self.iou_spinbox.setValue(x/100)        # The slider value changes, changing the box
            self.show_status('IOU Threshold: %s' % str(x/100))
            self.yolo_predict.iou_thres = x/100
        elif flag == 'conf_spinbox':
            self.conf_slider.setValue(int(x*100))
        elif flag == 'conf_slider':
            self.conf_spinbox.setValue(x/100)
            self.show_status('Conf Threshold: %s' % str(x/100))
            self.yolo_predict.conf_thres = x/100
        elif flag == 'speed_spinbox':
            self.speed_slider.setValue(x)
        elif flag == 'speed_slider':
            self.speed_spinbox.setValue(x)
            self.show_status('Delay: %s ms' % str(x))
            self.yolo_predict.speed_thres = x  # ms
            
    # change model
    def change_model(self,x):
        self.select_model = self.model_box.currentText()
        self.yolo_predict.new_model_name = "./models/%s" % self.select_model
        self.show_status('Change Model：%s' % self.select_model)
        self.Model_name.setText(self.select_model)

    # label result
    # def show_labels(self, labels_dic):
    #     try:
    #         self.result_label.clear()
    #         labels_dic = sorted(labels_dic.items(), key=lambda x: x[1], reverse=True)
    #         labels_dic = [i for i in labels_dic if i[1]>0]
    #         result = [' '+str(i[0]) + '：' + str(i[1]) for i in labels_dic]
    #         self.result_label.addItems(result)
    #     except Exception as e:
    #         self.show_status(e)

    # Cycle monitoring model file changes
    def ModelBoxRefre(self):
        pt_list = os.listdir('./models')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./models/' + x))
        # It must be sorted before comparing, otherwise the list will be refreshed all the time
        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.model_box.clear()
            self.model_box.addItems(self.pt_list)

    # Get the mouse position (used to hold down the title bar and drag the window)
    def mousePressEvent(self, event):
        p = event.globalPosition()
        globalPos = p.toPoint()
        self.dragPos = globalPos

    # Optimize the adjustment when dragging the bottom and right edges of the window size
    def resizeEvent(self, event):
        # Update Size Grips
        UIFuncitons.resize_grips(self)

    # Exit Exit thread, save settings
    def closeEvent(self, event):
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.iou_spinbox.value()
        config['conf'] = self.conf_spinbox.value()
        config['rate'] = self.speed_spinbox.value()
        config['save_res'] = (0 if self.save_res_button.checkState()==Qt.Unchecked else 2)
        config['save_txt'] = (0 if self.save_txt_button.checkState()==Qt.Unchecked else 2)
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        # Exit the process before closing
        if self.yolo_thread.isRunning():
            self.yolo_predict.stop_dtc = True
            self.yolo_thread.quit()
            MessageBox(
                self.close_button, title='Note', text='Exiting, please wait...', time=3000, auto=True).exec()
            sys.exit(0)
        else:
            sys.exit(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Home = MainWindow()
    Home.show()
    sys.exit(app.exec())  
