#encoding=gbk
import time
import h5py
import numpy as np
import os
def getMoves(iters,more_cubes):
    """
    :param iters: 迭代次数
    :param more_cubes: 7*6面板
    :return: 返回最佳的move
    """
    start_time=time.time()
    if more_cubes:
        rs=6
        cs=7
    else:
        rs=5
        cs=6
    all_move=[]
    count=0
    for r in range(rs):
        for c in range(cs):
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
                if move["steps"]==iters:
                    continue
                cur_r,cur_c=move['currentPoint']
                for dr,dc in [(-1,0),(0,1),(1,0),(0,-1)]:
                    if  not (cur_r + dr in range(rs) and cur_c + dc in range(cs)):
                        continue
                        #如果跑到了边界外面，那么不往下走
                    if  move['prePosition'] is not None and  \
                        ((cur_r + dr)==move['prePosition'][0] and (cur_c+dc)==move['prePosition'][1]):
                        count+=1
                        continue#如果有上一步，并且下一步是上一步，代表来回晃了晃，不要这样的情况。

                    # if move['steps']==0 and move['panel'][cur_r + dr,cur_c + dc]==move['panel'][cur_r,cur_c]:
                    #     count+=1
                    #     continue#第一步的时候如果下一步和当前色一样，不处理。因为别的出发点会处理这样的情况
                    # result_panel=swapArray2D(move['panel'],(cur_r,cur_c),(cur_r+dr,cur_c+dc))
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
    print("找出所有解耗时%s"%(time.time()-start_time))
    return all_move

    # start_time=time.time()
    # optimal_index,optimal_combo=getOptimal(np.array(all_panel))
    # print("求最优解耗时:",time.time()-start_time)
    # return len(all_move),all_move[optimal_index],optimal_combo,count
def saveMoves(iters,panel='panel_67'):
    # panel_56
    result = getMoves(iters, panel=='panel_67')
    start_points = []
    historys = []
    end_points=[]
    for mv in result:
        if mv["steps"] >= steps :
            start_points.append(mv['startPoint'])
            historys.append(mv["history"])
            end_points.append(mv["currentPoint"])
    os.makedirs(panel,exist_ok=True)
    f = h5py.File(f"{panel}/steps_{iters}.h5", 'w')
    f["start_points"] = start_points
    f["historys"] = historys
    f["end_points"]=end_points
    f.close()
if __name__ == '__main__':
    for steps in range(9,14):
        panel_type='panel_56'
        saveMoves(steps,panel_type)
