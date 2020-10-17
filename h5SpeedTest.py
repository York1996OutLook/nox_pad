#encoding=gbk
import time
import h5py
import numpy as np
import os
from config import Panel
def getMoves(p,percentage):
    """
    :return: 返回最佳的move
    """
    start_time=time.time()
    all_move=[]
    count=0
    for r in range(p.rows):
        for c in range(p.cols):
            start_r,start_c=r,c
            start_pos=dict()
            start_pos['startPoint']=start_r,start_c
            start_pos['currentPoint']=start_pos["startPoint"]
            start_pos['prePosition']=None
            start_pos['history']=[]
            start_pos['steps']=0
            panel_stack=[start_pos]
            while True:
                if len(panel_stack)==0:
                    break
                move=panel_stack.pop()
                if move["steps"]==p.steps:
                    continue
                cur_r,cur_c=move['currentPoint']

                for dr,dc in [(-1,0),(0,1),(1,0),(0,-1)]:
                    if  not (cur_r + dr in range(p.rows) and cur_c + dc in range(p.cols)):
                        #如果跑到了边界外面，那么不往下走
                        continue
                    if  move['prePosition'] is not None and (cur_r + dr,cur_c+dc)==move['prePosition']:
                        count+=1
                        # 如果有上一步，并且下一步是上一步，代表来回晃了晃，不要这样的情况。
                        continue

                    if np.random.randint(1,100)>percentage:
                        continue
                    #共5个属性
                    new_move=dict()
                    # new_move['panel']=result_panel#1
                    new_move['startPoint']=move['startPoint']#2
                    new_move['currentPoint']=(cur_r+dr,cur_c+dc)#3
                    new_move['prePosition']=cur_r,cur_c
                    his=move['history'].copy()
                    his.append((cur_r + dr, cur_c + dc))
                    new_move['history']=his#4
                    new_move['steps']=move['steps']+1#5

                    panel_stack.append(new_move)
                    # all_panel.append(new_move["panel"])
                    all_move.append(new_move)
    # print(f"找出所有解{len(all_move)}个,耗时{time.time()-start_time}")
    return all_move

    # start_time=time.time()
    # optimal_index,optimal_combo=getOptimal(np.array(all_panel))
    # print("求最优解耗时:",time.time()-start_time)
    # return len(all_move),all_move[optimal_index],optimal_combo,count
def saveMoves(p:Panel,percentage):
    # panel_56
    result = getMoves(p,percentage)
    start_points = []
    historys = []
    end_points=[]
    for mv in result:
        if mv["steps"] >= steps :
            start_points.append(mv['startPoint'])
            historys.append(mv["history"])
            end_points.append(mv["currentPoint"])
    save_dir=f'panel_{panel.panel_type}'
    os.makedirs(save_dir,exist_ok=True)
    f = h5py.File(f"{save_dir}/steps_{p.steps}.h5", 'w')
    f["start_points"] = start_points
    f["historys"] = historys
    f["end_points"]=end_points
    f.close()
    return len(historys)
if __name__ == '__main__':
    for steps in range(1,11):
        # for percent in range(1,100,10):
        percent = 90
        print(steps,end='\t')
        print(percent,end='\t')
        panel=Panel('56',steps,load_h5=False)
        sulutions=saveMoves(panel,percent)
        print(sulutions,end='\t\t')
        print()
