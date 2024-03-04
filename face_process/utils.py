def process_box(box):
    box = tuple(map(int, box))
    x0, y0 = box[0], box[1]
    x1, y1 = box[2], box[3]
    delta_x = x1 - x0
    delta_y = y1 - y0

    if delta_x == delta_y:
        return box
    if delta_x < delta_y:
        x0_ = x0
        x1_ = x1
        delta = delta_y - delta_x
        down = delta // 2
        up = delta - down
        y0_ = y0 + down
        y1_ = y1 - up

    else:
        y0_ = y0
        y1_ = y1
        delta = delta_x - delta_y
        right = delta // 2
        left = delta - right
        x0_ = x0 + right
        x1_ = x1 - left

    return [x0_, y0_, x1_, y1_]