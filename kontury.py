
import cv2
import numpy as np
import math
import csv

# (13.099, 123.16326530612243, 146.99999999999997) (RGB 147, 107, 76) brąz
# (19.184, 176.81603773584905, 212.0) (RGB 212, 159, 65) złoto
# (25.909, 22.530120481927717, 248.99999999999997) (RGB 249, 246, 227) beż


#sprawdzić filtry
class cnt:
    def klucz1(img):
        kernel1 = np.ones((12,12), np.uint8)
        wsp2=0.04
        out = img.copy()

        szary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #blur = cv2.GaussianBlur(szary, (5, 5), 0)
        edges = cv2.Canny(szary, 200, 255, apertureSize=3)
        laplacian = cv2.Laplacian(szary, cv2.CV_64F, ksize=9)

        ############################
        # # Definicja parametrów PDE
        # delta_t = 0.1
        # kappa = 10
        # iterations = 5
        # # Inicjalizacja funkcji gęstości
        # g = np.ones_like(szary) / np.sqrt(2)
        # # Iteracyjne wykonywanie PDE
        # for i in range(iterations):
        #     # Obliczenie gradientu
        #     grad_x, grad_y = np.gradient(szary)
        #
        #     # Obliczenie normy gradientu
        #     grad_norm = np.sqrt(grad_x ** 2 + grad_y ** 2)
        #
        #     # Obliczenie dywergencji
        #     div = (grad_x / (grad_norm + 1e-7)) + (grad_y / (grad_norm + 1e-7))
        #
        #     # Obliczenie funkcji gęstości
        #     g = 1 / (1 + (div / kappa) ** 2)
        #
        #     # Aktualizacja obrazu
        #
        #     print(g)
        #     szary2 = delta_t * g * div
        #     ret,thresh = cv2.threshold(szary2,0,10,cv2.THRESH_BINARY)
        #
        # # Wyświetlenie wyniku
        # cv2.imshow('Wynik0', szary2)
        # cv2.imshow('Wynik', thresh)
########################################
#################################
        # lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        #
        # # Narysowanie linii na obrazie
        # for line in lines:
        #     rho, theta = line[0]
        #     a = np.cos(theta)
        #     b = np.sin(theta)
        #     x0 = a * rho
        #     y0 = b * rho
        #     x1 = int(x0 + 1000 * (-b))
        #     y1 = int(y0 + 1000 * (a))
        #     x2 = int(x0 - 1000 * (-b))
        #     y2 = int(y0 - 1000 * (a))
        #     cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        #
        # # Wyświetlenie obrazu z zaznaczonymi krawędziami
        # cv2.imshow('image', img)
#####################################
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        fill_img = cv2.drawContours(edges.copy(), contours, -1, (255, 255, 255), thickness=cv2.FILLED)

        cv2.imshow("fill", fill_img)
        cv2.imshow("canny", edges)
        cv2.imshow("laplaciany", laplacian)






        #closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel1)


        # center_list = []
        # kontury, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # detekcja konturów
        # for k in kontury:
        #     x, y, w, h = cv2.boundingRect(k)
        #     if (w > 50 and h > 50) and abs(1-w/h)<0.5:
        #         if x == 0 and y == 0:
        #             continue
        #         rogi = cv2.approxPolyDP(k, wsp2 * cv2.arcLength(k, True), True)  # pozyskiwanie ilosci rogów
        #         if len(rogi) == 4:
        #             angles = cnt.calculate_angles(rogi[0][0], rogi[1][0], rogi[2][0], rogi[3][0])
        #
        #             if((abs(90-angles[0])<10) and (abs(90-angles[1])<10) and (abs(90-angles[2])<10) and (abs(90-angles[3])<10)):
        #                 cv2.drawContours(out, [rogi], 0, (255, 0, 0), 2)  # rysowanie konturów
        #                 cv2.putText(out, "kwadrat", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)  # wstawianie tekstu
        #                 center_x = int(x + w / 2)
        #                 center_y = int(y + h / 2)
        #                 center = (center_x, center_y)
        #                 center_list.append(center)
        #                 cv2.circle(out, center, 5, (0, 0, 255), -1)
        #
        # cv2.imshow("kwadraty1", out)
        #return center_list

    def klucz2(img):
        out = img.copy()
        wsp2 = 0.001
        kernel1 = np.ones((5, 5), np.uint8)
        kernel2 = np.ones((1, 1), np.uint8)
        kernel3 = np.ones((9, 9), np.uint8)

        szary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #szary2 = cv2.equalizeHist(szary)
        thresh = cv2.adaptiveThreshold(szary, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 20)


        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel2)
        cv2.imshow("thresh", thresh)
        cv2.imshow("szary1", szary)



        # blur = cv2.GaussianBlur(szary, (5, 5), 0)
        edges = cv2.Canny(thresh, 220, 255, apertureSize=3)
        cv2.imshow("edges", edges)

        mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel1)


        fill_img = mask.copy()
        contours, hierarchy = cv2.findContours(fill_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Generowanie obrazu z konturami
        contour_img = np.zeros_like(fill_img)
        cv2.drawContours(contour_img, contours, -1, 255, 1)

        # Wypełnienie obszaru w środku każdego konturu
        for cnt in contours:
            # Wyznaczanie wierzchołków konturu
            pts = cnt.reshape((-1, 1, 2)).astype(np.int32)

            # Wypełnianie obszaru w środku konturu
            cv2.fillPoly(contour_img, [pts], 255)


        thresh = cv2.morphologyEx(contour_img, cv2.MORPH_OPEN, kernel3)
        cv2.imshow("fill_img", thresh)
        #cv2.imshow("mask", mask)



        kontury, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # detekcja konturów
        for k in kontury:
            rogi = cv2.approxPolyDP(k, wsp2 * cv2.arcLength(k, True), True)  # pozyskiwanie ilosci rogów
            cv2.drawContours(out, [rogi], 0, (255, 0, 0), 2)  # rysowanie konturów
            for corner in rogi:
                x, y = corner.ravel()
                cv2.circle(out, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow("kwadraty2", out)

    def antiflash(img):
        out = img.copy()

        img_float = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0

        # utworzenie obiektu tonemappera
        tonemap = cv2.createTonemap(gamma=1.0)

        # zastosowanie funkcji tonowania obrazu
        img_tonemapped = tonemap.process(img_float)

        # zapisanie wyniku do pliku
        out = img_tonemapped
        out = result_img = cv2.convertScaleAbs(out*255)

        return out



    def klucz3(img):
        out = img.copy()
        wsp2 = 0.001
        kernel1 = np.ones((30, 30), np.uint8)
        kernel2 = np.ones((3, 3), np.uint8)
        kernel3 = np.ones((9, 9), np.uint8)



        cv2.imshow("edge canny", cv2.Canny(img, 150, 255, apertureSize=3))

        #cv2.imshow("img", img)
        blur = cv2.bilateralFilter(img, 9, 140, 10)
        #cv2.imshow("blur", blur)




        # blur = cv2.GaussianBlur(szary, (5, 5), 0)
        edges = cv2.Canny(blur, 150, 255, apertureSize=3)
        #cv2.imshow("edges", edges)

        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel3)

        min_element_length = 50

        # stwórz maskę o wymiarach takich samych jak obraz wyjściowy z cv2.Canny(), wypełnioną zerami
        mask = np.zeros_like(edges)

        # znajdź kontury w obrazie wyjściowym z cv2.Canny()
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # dla każdego konturu odrzuć te, które są krótsze niż minimalna akceptowalna długość
        for contour in contours:
            length = cv2.arcLength(contour, True)
            if length >= min_element_length:
                cv2.drawContours(mask, [contour], -1, 255, -1)

        result = cv2.bitwise_and(edges, edges, mask=mask)
        cv2.imshow('result', result)
        #result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)






        mask = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel1)
        cv2.imshow("mask", mask)

        fill_img = mask.copy()
        contours, hierarchy = cv2.findContours(fill_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Generowanie obrazu z konturami
        contour_img = np.zeros_like(fill_img)
        cv2.drawContours(contour_img, contours, -1, 255, 1)

        # Wypełnienie obszaru w środku każdego konturu
        for cnt in contours:
            # Wyznaczanie wierzchołków konturu
            pts = cnt.reshape((-1, 1, 2)).astype(np.int32)

            # Wypełnianie obszaru w środku konturu
            cv2.fillPoly(contour_img, [pts], 255)


        thresh = cv2.morphologyEx(contour_img, cv2.MORPH_OPEN, kernel3)
        cv2.imshow("fill_img", thresh)
        #cv2.imshow("mask", mask)



        kontury, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # detekcja konturów
        for k in kontury:
            length = cv2.arcLength(k, True)
            if length>200:
                rogi = cv2.approxPolyDP(k, wsp2 * cv2.arcLength(k, True), True)  # pozyskiwanie ilosci rogów
                cv2.drawContours(out, [rogi], 0, (255, 0, 0), 2)  # rysowanie konturów
                for corner in rogi:
                    x, y = corner.ravel()
                    cv2.circle(out, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow("kwadraty2", out)


    def oczkoDetection(img):
        out = img.copy()
        wsp2 = 0.001
        kernel1 = np.ones((55, 55), np.uint8)
        kernel2 = np.ones((3, 3), np.uint8)
        kernel3 = np.ones((9, 9), np.uint8)

        cv2.imshow("edge canny", cv2.Canny(img, 150, 255, apertureSize=3))

        # cv2.imshow("img", img)
        blur = cv2.bilateralFilter(img, 9, 140, 10)
        # cv2.imshow("blur", blur)

        # blur = cv2.GaussianBlur(szary, (5, 5), 0)
        edges = cv2.Canny(blur, 150, 255, apertureSize=3)
        # cv2.imshow("edges", edges)

        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel3)

        min_element_length = 50

        # stwórz maskę o wymiarach takich samych jak obraz wyjściowy z cv2.Canny(), wypełnioną zerami
        mask = np.zeros_like(edges)

        # znajdź kontury w obrazie wyjściowym z cv2.Canny()
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # dla każdego konturu odrzuć te, które są krótsze niż minimalna akceptowalna długość
        for contour in contours:
            length = cv2.arcLength(contour, True)
            if length >= min_element_length:
                cv2.drawContours(mask, [contour], -1, 255, -1)

        result = cv2.bitwise_and(edges, edges, mask=mask)
        cv2.imshow('result', result)
        # result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        mask = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel1)
        cv2.imshow("mask", mask)

        #
        #
        #
        center_list=[]
        contours, _ = cv2.findContours(result, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for k in contours:
            (x, y), radius = cv2.minEnclosingCircle(k)
            if 400 > radius > 40 :
                # obliczenie współczynników kształtu (momenty)
                moments = cv2.moments(k)
                if moments['m00'] != 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                    circularity = 4 * np.pi * moments['m00'] / (cv2.arcLength(k, True) ** 2)
                    if circularity > 0.8:
                        # narysowanie okręgu na obrazie
                        cv2.circle(out, (cx, cy), int(radius), (0, 255, 0), 2)
                        cv2.rectangle(out, (cx, cy), (cx + 5, cy + 5), (0, 128, 255), -1)
                        center_list.append((cx, cy))

        cv2.imshow("final", out)




        # center_list = []
        # circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 200, param1=50, param2=30, minRadius=30, maxRadius=300) #dostosować parametry
        # if circles is not None:
        #     circles = np.round(circles[0, :]).astype("int")
        #     for (x, y, r) in circles:
        #         cv2.circle(out, (x, y), r, (0, 255, 0), 2)
        #         cv2.rectangle(out, (x, y), (x + 5, y + 5), (0, 128, 255), -1)
        #         center_list.append((x, y))
        #
        # cv2.imshow("final", out)
        return center_list


    def distance_calibration(wysokosc, h_pix, w_pix, h_r, w_r):
        wsp = ((h_pix+w_pix)/2 * wysokosc)/((h_r+w_r)/2)
        return wsp

        # na podstawie samych masek

    def calibration_help(img):
        wsp2 = 0.04
        out = img.copy()

        szary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(szary, 200, 255, cv2.THRESH_BINARY)
        blur = cv2.GaussianBlur(mask, (5, 5), 0)

        kontury, _ = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # detekcja konturów
        center_list = []
        center=[]
        average=0
        for k in kontury:
            x, y, w, h = cv2.boundingRect(k)
            if (w > 50 and h > 50) and abs(1 - w / h) < 0.5:
                if x == 0 and y == 0:
                    continue
                rogi = cv2.approxPolyDP(k, wsp2 * cv2.arcLength(k, True), True)  # pozyskiwanie ilosci rogów
                if len(rogi) == 4:
                    angles = cnt.calculate_angles(rogi[0][0], rogi[1][0], rogi[2][0], rogi[3][0])
                    kopia_rogi=rogi

                    if ((abs(90 - angles[0]) < 10) and (abs(90 - angles[1]) < 10) and (abs(90 - angles[2]) < 10) and (abs(90 - angles[3]) < 10)):
                        cv2.drawContours(out, [rogi], 0, (255, 0, 0), 2)  # rysowanie konturów
                        cv2.putText(out, "kwadrat", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50),2)  #wstawianie tekstu
                        center_x = int(x + w / 2)
                        center_y = int(y + h / 2)
                        center = (center_x, center_y)
                        cv2.circle(out, center, 5, (0, 0, 255), -1)

                        #######
                        distance1 = np.linalg.norm(np.array(kopia_rogi[0][0]) - np.array(kopia_rogi[1][0]))
                        distance2 = np.linalg.norm(np.array(kopia_rogi[1][0]) - np.array(kopia_rogi[2][0]))
                        distance3 = np.linalg.norm(np.array(kopia_rogi[2][0]) - np.array(kopia_rogi[3][0]))
                        distance4 = np.linalg.norm(np.array(kopia_rogi[3][0]) - np.array(kopia_rogi[0][0]))
                        average = (distance1 + distance2 + distance3 + distance4) / 4
                        average1= (distance1 + distance3) / 2
                        average2 = (distance2 + distance4) / 2

                        print(distance1, distance2, distance3, distance4)
                        print(average1, average2, average)

        cv2.imshow("kwadraty2", out)
        return center, average1, average2, average


    def calculate_angles(vertex1, vertex2, vertex3, vertex4):
        # Oblicz długości boków
        try:
            edge1 = math.sqrt((vertex2[0] - vertex1[0]) ** 2 + (vertex2[1] - vertex1[1]) ** 2)
            edge2 = math.sqrt((vertex3[0] - vertex2[0]) ** 2 + (vertex3[1] - vertex2[1]) ** 2)
            edge3 = math.sqrt((vertex4[0] - vertex3[0]) ** 2 + (vertex4[1] - vertex3[1]) ** 2)
            edge4 = math.sqrt((vertex1[0] - vertex4[0]) ** 2 + (vertex1[1] - vertex4[1]) ** 2)
            angle1 = math.degrees(math.acos((edge2 ** 2 + edge4 ** 2 - edge1 ** 2 - edge3 ** 2) / (2 * edge2 * edge4)))
            angle2 = math.degrees(math.acos((edge3 ** 2 + edge1 ** 2 - edge2 ** 2 - edge4 ** 2) / (2 * edge3 * edge1)))
            angle3 = math.degrees(math.acos((edge2 ** 2 + edge4 ** 2 - edge1 ** 2 - edge3 ** 2) / (2 * edge2 * edge4)))
            angle4 = math.degrees(math.acos((edge3 ** 2 + edge1 ** 2 - edge2 ** 2 - edge4 ** 2) / (2 * edge3 * edge1)))
        except ValueError:
            angle1 = 0
            angle2 = 0
            angle3 = 0
            angle4 = 0

        return (angle1, angle2, angle3, angle4)

    def distortion_reduction(img):
        calibration_matrix  = np.load("calibration_matrix.npy")
        distortion_coefficients = np.load("distortion_coefficients.npy")
        undistorted_img = cv2.undistort(img, calibration_matrix, distortion_coefficients)
        cv2.imshow('Obraz bez dystorsji', undistorted_img)













