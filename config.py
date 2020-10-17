# encoding=utf-8
# @Time : 2020/10/14 21:49
# @Author : qqyor
# @QQ : 603997262
# @File : configs.py
# @Project : Nox_pad
from easydict import EasyDict
import h5py
from benchmark import showtime
class Panel:
    def __init__(self,panel_type,steps):
        self.res_w=540
        self.res_h=960
        self.panel_type=panel_type
        self.start_row=514
        self.end_row=954
        self.start_col=6
        self.end_col=534
        self.start_points=None
        self.historys=None
        self.end_points=None

        self.ball_w=88
        self.ball_h=88
        self.steps=steps
        if self.panel_type=='56':
            self.rows=5
            self.cols=6
        else:   # self.panel_type='67'
            self.rows=6
            self.cols=7

        self.panel_w=self.end_col-self.start_col
        self.panel_h=self.end_row-self.start_row

        self.load_h5(steps)
    @showtime
    def load_h5(self,steps):
        f = h5py.File(f"panel_{self.panel_type}/steps_{steps}.h5", 'r')
        self.start_points = f["start_points"][()]
        self.historys = f["historys"][()]
        self.end_points = f["end_points"][()]
        f.close()
