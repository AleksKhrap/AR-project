import numpy as np
import cv2
import os
import requests
from bs4 import BeautifulSoup


class FootballClubs:
    def __init__(self):
        """Инициализирует переменные"""
        self.MIN_MATCH_COUNT = 15
        self.FLANN_INDEX_LSH = 6
        self.Y0, self.DY = 20, 20

    @staticmethod
    def parse(club_name):
        """Собирает статистику клуба"""
        url = 'https://premierliga.ru/about-rpl/clubs/'
        if club_name == 'pfc-cska':
            url += 'pfc-cska'
        if club_name == 'lokomotiv':
            url += 'lokomotiv'
        if club_name == 'spartak':
            url += 'spartak'
        if club_name == 'zenit':
            url += 'zenit'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        a = soup.select('.count p')
        a1 = a[1].text
        a2 = a[2].text
        a3 = a[3].text
        a4 = a[4].text
        return [a1, a2, a3, a4]

    def object_highlighting(self):
        """Выделяет эмблему и выводит статистику"""
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        images = sorted(os.listdir('data_clubs'))

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Stream end.")
                break
            orb = cv2.ORB_create()

            try:
                for filename in images:
                    img = cv2.imread(os.path.join('data_clubs', filename))
                    kp1, des1 = orb.detectAndCompute(img, None)
                    kp2, des2 = orb.detectAndCompute(frame, None)

                    index_params = dict(algorithm=self.FLANN_INDEX_LSH,
                                        table_number=6,
                                        key_size=12,
                                        multi_probe_level=2)
                    search_params = dict(checks=50)
                    flann = cv2.FlannBasedMatcher(index_params, search_params)
                    matches = flann.knnMatch(des1, des2, k=2)

                    good = []
                    try:
                        for m, n in matches:
                            if m.distance < 0.7 * n.distance:
                                good.append(m)
                    except ValueError:
                        cv2.imshow("RPL", frame)

                    if len(good) > self.MIN_MATCH_COUNT:
                        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        h, w = img.shape[:2]
                        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, M)

                        if filename == images[0]:
                            frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
                            stat = self.parse('pfc-cska')
                            text = f'CSKA\n' \
                                   f'Games: {str(stat[0])}\n' \
                                   f'Wins: {str(stat[1])}\n' \
                                   f'Draws: {str(stat[2])}\n' \
                                   f'Defeats: {str(stat[3])}'
                            for i, line in enumerate(text.split('\n')):
                                y1 = self.Y0 + i * self.DY
                                cv2.putText(frame, line, (20, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.imshow("RPL", frame)

                        if filename == images[1]:
                            frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                            y, x = frame.shape[:2]
                            stat = self.parse('lokomotiv')
                            text = f'Lokomotiv\n' \
                                   f'Games: {str(stat[0])}\n' \
                                   f'Wins: {str(stat[1])}\n' \
                                   f'Draws: {str(stat[2])}\n' \
                                   f'Defeats: {str(stat[3])}'
                            for i, line in enumerate(text.split('\n')):
                                y1 = self.Y0 + i * self.DY
                                cv2.putText(frame, line, (x - 150, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.imshow("RPL", frame)

                        if filename == images[2]:
                            frame = cv2.polylines(frame, [np.int32(dst)], True, (255, 255, 255), 3, cv2.LINE_AA)
                            y, x = frame.shape[:2]
                            stat = self.parse('spartak')
                            text = f'Spartak\n' \
                                   f'Games: {str(stat[0])}\n' \
                                   f'Wins: {str(stat[1])}\n' \
                                   f'Draws: {str(stat[2])}\n' \
                                   f'Defeats: {str(stat[3])}'
                            for i, line in enumerate(text.split('\n')):
                                y1 = self.Y0 - i * self.DY
                                cv2.putText(frame, line, (x - 150, y - y1 - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                            (255, 255, 255), 2)
                            cv2.imshow("RPL", frame)

                        if filename == images[3]:
                            frame = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)
                            y, x = frame.shape[:2]
                            stat = self.parse('zenit')
                            text = f'Zenit\n' \
                                   f'Games: {str(stat[0])}\n' \
                                   f'Wins: {str(stat[1])}\n' \
                                   f'Draws: {str(stat[2])}\n' \
                                   f'Defeats: {str(stat[3])}'
                            for i, line in enumerate(text.split('\n')):
                                y1 = self.Y0 - i * self.DY
                                cv2.putText(frame, line, (20, y - y1 - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
                                            2)
                            cv2.imshow("RPL", frame)
                    else:
                        cv2.imshow("RPL", frame)

            except cv2.error:
                cv2.imshow("RPL", frame)

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


FootballClubs().object_highlighting()
