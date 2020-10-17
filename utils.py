# encoding=utf-8
# @Time : 2020/10/6 19:45
# @Author : qqyor
# @QQ : 603997262
# @File : utils.py
# @Project : Nox_pad
import h5py
import os
import numpy as np
from  PIL import Image
import time
from functools import wraps
import cv2
from  config import Panel
from numba import jit
from benchmark import showtime
from skimage.metrics import structural_similarity

@jit()
def up_down(aggregation,checkerboard):
    num, rs, cs = aggregation.shape

    result = -np.ones((num, rs, cs), np.int)
    # 三连的数字上浮为0，保留的字数字下沉
    # checkerboard[aggregation==True]=-1

    for n in range(num):
        for r in range(rs):
            for c in range(cs):
                if aggregation[n, r, c]:
                    continue
                result[n, r + np.sum(aggregation[n, r + 1:, c]), c] = checkerboard[n, r, c]
    return result
@jit()
def getAggregation(diff1,diff2,diff3,diff4,checkerboard):
    num, rs, cs = checkerboard.shape

    aggregation = np.zeros((num, rs, cs), np.bool)
    aggregation[checkerboard == -1] = True
    down_diff= (diff1 == 0) & (diff2 == 0)
    horizontal_diff=(diff3 == 0) & (diff4 == 0)

    aggregation[:, :-1, :] |= down_diff
    aggregation[:, 1:, :] |= down_diff
    aggregation[:, 2:, :] |= down_diff[:, :-1, :]

    aggregation[:, :, :-1] |= horizontal_diff
    aggregation[:, :, 1:] |= horizontal_diff
    aggregation[:, :, 2:] |= horizontal_diff[...,:-1]
    return aggregation
@jit()
def getOptimal(_panel):
    # _panel = np.random.randint(1, 5, (2, 5, 6))
    # _panel[:, 1:5, 2] = 1
    # _panel[:, 2, 1:5] = 1

    checkerboard=_panel.copy()
    num, rs, cs = checkerboard.shape
    pre_score=np.zeros(num,np.int)
    keep_going=list()#key:value : cur:pre
    # keep_going.append({index:index for index in range(num)})
    max_score=-1
    max_score_index=-1
    iter_num=0
    max_score_location=iter_num
    while True:
        num, rs, cs = checkerboard.shape
        diff1 = np.diff(checkerboard, 1, 1)
        diff2 = np.diff(checkerboard, 2, 1, append=7)  # 行和行相减
        # 这里为什么要补上6呢，因为6不会和其他的元素重复，不会产生错误的解，如果补的元素和已有元素刚好拼成了三个就会得一分
        diff3 = np.diff(checkerboard, 1, 2)
        diff4 = np.diff(checkerboard, 2, 2, append=7)  # 列和列相减
        # _score = ((diff1 == 0) & (diff2 == 0)).sum((1, 2)) + ((diff3 == 0) & (diff4 == 0)).sum((1, 2))
        aggregation=getAggregation(diff1,diff2,diff3,diff4,checkerboard)#三连的位置用True替换
        result=up_down(aggregation,checkerboard)#得到所有元素下降的结果
        # 聚合得到所有被消除的位置

        # 个数，行，列
        # skyfall = np.zeros((num, rs, cs), np.bool)
        # true上浮，false下沉
        # for n in range(num):
        #     for c in range(cs):
        #         skyfall[n, :np.sum(aggregation[n, :, c],0), c] = True



                #result[n, np.array([r + np.sum(aggregation[n, r + 1:, :], 0) for r in range(rs)]), range(cs)] = checkerboard[n]

        #如果开始就是
        # print(np.sum(aggregation, (1, 2)).argmax())
        # print(np.sum(aggregation, (1, 2)).max())

        checkerboard=[]

        cur_score=np.sum(aggregation,(1,2))#当前得分
        cur_max_score=cur_score.max()
        if cur_max_score>=max_score:#等于也更新，这样利于增加combo,也可以去掉
            max_score=cur_max_score
            max_score_index=cur_score.argmax()
            max_score_location=iter_num#迭代次数和
        cur_pre=dict()

        better_number=0
        if iter_num==0:
            for index in range(num):
                if cur_score[index]>=1:
                    cur_pre[better_number]=index
                    better_number+=1
                    checkerboard.append(result[index])
        else:
            for index in range(num):

                if cur_score[index]>pre_score[keep_going[len(keep_going)-1][index]]:#代表这个下落后产生了新的combo
                    cur_pre[better_number]=index#n对应m，m个中有n个会产生新的combo，n<m
                    better_number+=1
                    checkerboard.append(result[index])
        checkerboard=np.array(checkerboard)
        if better_number==0:
            break
        else:
            keep_going.append(cur_pre)#list of dict
            pre_score=cur_score
            # print("可以继续的数量：" , better_number)
            iter_num += 1

            continue

    # minist_index=np.sum(aggregation,(1,2)).argmax()
    # print("最佳迭代位置",max_score_location)
    max_index=max_score_index
    for dic in keep_going[:max_score_location][::-1]:
        max_index=dic[max_index]
    # max_value=np.sum(aggregation, (1, 2)).max()

    # minist_index=np.sum(aggregation,(1,2)).argmax()
    #
    # max_index=minist_index
    # for dic in keep_going[::-1]:
    #     max_index=dic[max_index]
    # max_value=np.sum(aggregation, (1, 2)).max()

    return max_index, max_score,_panel[max_index]

    # return _score.argmax(),_score.max()
def swapArray2D(arr,p1,p2):
    data=arr.copy()
    temp=data[p1]
    data[p1]=data[p2]
    data[p2]=temp
    return data
def walk(init_balls,start_positions,historys):

    # if arr[start_pos[0],start_pos[1]]==arr[paths[0][0],paths[0][1]]:
    #     return None
    steps=historys.shape[1]
    data=init_balls.copy()
    temp=np.empty((len(data)),dtype=np.int32)
    for i in range(len(data)):
        temp[i] = data[i,start_positions[i,0],start_positions[i,1]]
    for i in range(len(data)):
        data[i,start_positions[i,0],start_positions[i,1]]=data[i,historys[i,0,0],historys[i,0,1]]

    #左边的shape应该是
    for step in range(steps-1):
        for i in range(len(data)):
            data[i,historys[i,step,0],historys[i,step,1]]=data[i,historys[i,step+1,0],historys[i,step+1,1]]

    for i in range(len(data)):
        data[i,historys[i,steps-1,0],historys[i,steps-1,1]]=temp[i]
    return data

def showtime(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result= func(*args, **kwargs)
        end_time=time.time()
        print(f'{func.__name__} use time： {end_time - start_time} S')
        return result
    return wrapper

@jit
def getMovesFromH5(init_balls,p:Panel,start_filter=None):
    """
       :param init_balls: 初始面板
       :param p:当前面板类型
       :param start_filter
       :return: 返回最佳的move
       """

    # start_time = time.time()
    if start_filter is not None:
        start_points=p.start_points[start_filter]
        historys=p.historys[start_filter]
    else:
        start_points=p.start_points
        historys=p.historys

    # 按照h5文件中记录的history来移动init_balls
    all_panel=walk(np.expand_dims(init_balls,0).repeat(len(start_points),axis=0),start_points,historys)

    # print("找出所有解耗时%s" % (time.time() - start_time))

    # start_time = time.time()
    optimal_index, optimal_combo ,best_balls= getOptimal(all_panel)
    # print("求最优解耗时:", time.time() - start_time)
    # print(f"共消除了{optimal_combo}个")
    end_point=historys[optimal_index][-1]
    return len(historys),start_points[optimal_index],historys[optimal_index], best_balls, optimal_combo,end_point

@showtime
def generate_nox_script(p:Panel, best_start_point, all_path):
    """
    生成nox可以识别的动作序列
    :param p: panel
    :param best_start_point: point
    :param all_path: path
    :return:
    """
    root='C:/Users/qqyor/AppData/Local/Nox/record'
    file='1d6f68c2a6ab4293a2dc575efdbf1357'
    def mouse_down(x,y,ms)->str:
        return f'0ScRiPtSePaRaToR{p.res_w}|{p.res_h}|MULTI:1:0:{x}:{y}ScRiPtSePaRaToR{ms}'
    def slide(x,y,ms)->str:
        return f'0ScRiPtSePaRaToR{p.res_w}|{p.res_h}|MULTI:1:2:{x}:{y}ScRiPtSePaRaToR{ms}'
    def mouse_up(ms)->str:
        return f'0ScRiPtSePaRaToR{p.res_w}|{p.res_h}|MSBRL:0:0ScRiPtSePaRaToR{ms}'
    interval=50    # ms 间隔
    move_count=1
    row,col=best_start_point
    start_x=p.start_col+p.ball_w*(col+0.5)
    start_y=p.start_row+p.ball_h*(row+0.5)
    start_x,start_y=int(start_x),int(start_y)
    script=mouse_down(start_x,start_y,interval * move_count)+'\n'
    for point in all_path:
        move_count+=1
        row, col = point
        start_x = p.start_col + p.ball_w * (col + 0.5)
        start_y = p.start_row + p.ball_h * (row + 0.5)
        start_x, start_y = int(start_x), int(start_y)
        script += slide(start_x, start_y, interval * move_count)+'\n'
    move_count+=1
    script+=mouse_up(interval * move_count)
    open(f'{root}/{file}','w').write(script)
    print('done')
@showtime
def screen_shot():
    # nox截图
    # 截图并保存到01.txt
    nox_screen_shot_commnd='nox_adb shell screencap -p /sdcard/01.png'
    os.system(nox_screen_shot_commnd)
    cp_img2pc_command='nox_adb pull /sdcard/01.png E:/python/Nox_pad'
    os.system(cp_img2pc_command)

@showtime
def get_balls(gt_dict:dict,p,mask=None,shot:bool=True,cats=6):
    if shot:
        screen_shot()
    # 获取球球们
    img=cv2.imread('01.png')
    balls_region = img[p.start_row:p.end_row, p.start_col:p.end_col]

    types = np.empty((p.rows, p.cols),dtype=np.int)

    for row in range(p.rows):
        for col in range(p.cols):
            start_row,end_row=row*p.ball_h,(row+1)*p.ball_h
            start_col,end_col=col*p.ball_w,(col+1)*p.ball_w

            ball=balls_region[start_row:end_row,start_col:end_col].copy()
            # bg_mask=get_bg_mask()
            # balls_region[bg_mask.T] = 0
            ball[~mask] = 0
            cv2.imwrite(f'balls/{row}{col}.png',ball)
            cls=get_class(ball,gt_dict,classes=cats,)
            types[row,col]=cls
    return types

def get_class(ball:np.ndarray,gt_dict:dict,classes=6,):
    similarities=[]
    for key in range(classes):
        gt=gt_dict[key]
        # similarity=similarity_by_hist(gt,ball)
        similarity=structural_similarity(gt,ball,multichannel=True)
        similarities.append(similarity)
    return np.argmax(similarities)

@showtime
def get_all_type_img(mask,classes:int=6,)->dict:
    root='classes'
    gt_dict=dict()
    for cls in range(classes):
        img_file=f'{root}/{cls}.png'
        img=cv2.imread(img_file)
        img[~mask]=0
        gt_dict[cls]=img
    return gt_dict


def similarity_by_hist(image1, image2):
    # RGB每个通道的直方图相似度
    # 将图像resize后，分离为RGB三个通道，再计算每个通道的相似值

    hist1, _ = np.histogram(image1[..., 2], 256, [0, 255])
    hist2,_ = np.histogram(image2[..., 2], 256, [0, 255])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + \
                     (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)

    return degree
def get_mask(width=88,height=88,radius=40):
    grid = np.meshgrid(range(width), range(height))
    return (grid[0]-width/2)**2+(grid[1]-height//2)**2<radius**2

def get_bg_mask():
    mask=get_mask()
    bg=np.hstack([mask for _ in range(5)])
    bg_mask=np.vstack([bg.copy() for _ in range(6)])
    return bg_mask
    # arr=np.zeros()
    # for r in range(p.rows):
    #     for col in range(p.cols):

if __name__ == '__main__':
    getMovesFromH5(None,Panel('56',3),None)

