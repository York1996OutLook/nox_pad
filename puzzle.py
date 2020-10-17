# encoding=utf-8
# @Time : 2020/10/14 21:04
# @Author : qqyor
# @QQ : 603997262
# @File : puzzle.py
# @Project : Nox_pad
from utils import *
import config
from config import Panel
@jit()
def doPuzzle(cur_balls,cur_panel):
    all_path = []

    num_solution, best_start_position, best_history, best_balls, best_combo, end_position = getMovesFromH5(cur_balls,cur_panel)
    all_path.extend(best_history)
    best_start_point = best_start_position

    for _ in range(3):

        start_filter = ((cur_panel.start_points[:, 0] == end_position[0]) & (cur_panel.start_points[:, 1] == end_position[1]))

        num_solution, best_start_position, best_history, best_balls, best_combo, end_position = getMovesFromH5(
            best_balls, cur_panel,start_filter)

        all_path.extend(best_history)

    # print(f"一共{num_solution}种解决方案")
    # print(best_balls) if best_balls is not None else print(None)
    # print(f"走了{len(all_path)}步数")

    generate_nox_script(cur_panel, best_start_point, all_path)
if __name__ == '__main__':
    import utils

    cats=len(os.listdir('classes'))
    panel=Panel(panel_type='56',steps=9)
    valid_mask=get_mask()
    gt_imgs=get_all_type_img(classes=cats,mask=valid_mask)

    while True:
        balls = get_balls(gt_dict=gt_imgs,
                          p=panel,
                          mask=valid_mask,
                          shot=False,
                          cats=cats,)
        print(balls)
        doPuzzle(balls, panel)
        print(1)
