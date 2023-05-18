import cv2
import numpy as np
from kontury import cnt
#from pose_estimation import estimation
import time
import keyboard

#ręczna kalibracja kolorów
# drzewo.zapisywanie("braz0", np.array([12, 123, 147]))
#
rozmiar=(1280, 720)
# # #
# #### ze zdjęcia
# img = cv2.imread("klucz1_3.jpg")
# frame = cv2.resize(img, rozmiar)
#
# cv2.imshow("orginal", frame)
# cnt.klucz2(frame)
#
#
# #tonned = cnt.antiflash(frame)
# #cv2.imshow("tonned", tonned)
# #cnt.klucz2(tonned)
#
# #
# #
# #
# #
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# #
# cap = cv2.VideoCapture('14.mp4')
#
# lista_wsp=[]
#
#
# while cap.isOpened():    # wczytanie ramki
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # konwersja kolorów
#     frame = cv2.resize(frame, rozmiar)
#     out = frame.copy()
#     cv2.imshow("orginal", frame)
#
#     #### kalibracja na kliknięcie
#     #cv2.setMouseCallback('orginal', drzewo.save_hsv, param=(frame, "zloto"))
#     #################### kalibracka wsp odległości
#     #center, avg = drzewo.calibration_help((frame))
#     # if(avg!=0):
#     #     lista_wsp.append(estimation.distance_calibration(1400, avg, avg, 100, 100))
#     # #########
#
#     center_list = drzewo.kwadraty3(frame)
#     #center_list = drzewo.kola1(frame, "zloto")
#     if(len(center_list)!=0):
#         center, avg = drzewo.calibration_help((frame))
#         x, y = estimation.calculate_xy(center, rozmiar, 14000, estimation.wsp0)
#         print(estimation.to_gps(x, y, 0))
#
#     # for center in center_list:
#     #      estimation.calculate_xy(center)
#
#
# ##### mechanizm pauzy
#     if keyboard.is_pressed('s'):
#         time.sleep(0.5)
#         while True:
#             if keyboard.is_pressed('s'):
#                 time.sleep(0.5)
#                 break
#
#     time.sleep(0.1)
#     if cv2.waitKey(1) == ord('q'):
#         break
#
#
#
#
#
#
# # #print(np.mean(lista_wsp))




# center, avg = drzewo.calibration_help((hd_image))
#
# x, y = estimation.calculate_xy(center, rozmiar, 14000, estimation.wsp0)
# print(estimation.to_gps(x, y, 0))



camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    #frame = cv2.resize(frame, rozmiar)

    cv2.imshow("orginal", frame)
    #cnt.klucz3(frame)
    #cnt.oczkoDetection(frame)
    #cnt.distortion_reduction(frame)

    if cv2.waitKey(1) == 27: # naciśnięcie klawisza ESC kończy program
        break


camera.release()
cv2.destroyAllWindows()

