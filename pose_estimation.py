import math

ratio =230

class estimation:
    # zwrca przesunięcie względem osi widzenia, dodatnie x -> w prawo, dodatnie y -> w dół
    def calculate_xy(center, rozmiar, high, wsp):
        ratio = high / wsp  # cm/pixel

        cx = center[0]
        cy = center[1]
        cx = cx - rozmiar[0] / 2
        cy = cy - rozmiar[1] / 2

        x = ratio * cx
        y = ratio * cy
        print(x, y)
        return x, y

    def distance_calibration(wysokosc, h_pix, w_pix, h_r, w_r):
        wsp = ((h_pix+w_pix)/2 * wysokosc)/((h_r+w_r)/2)
        return wsp