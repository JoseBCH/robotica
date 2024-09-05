# Librerias
import cv2
from ultralytics import YOLO
import math

class ShopIA:
    def __init__(self):
        # VideoCapture
        self.cap = cv2.VideoCapture(1)
        self.cap.set(3, 1200)
        self.cap.set(4, 700)

        # MODELOS:
        self.ObjectModel = YOLO('C:/Tienda/modelos/yolov8l.pt')
        self.billModel = YOLO('C:/Tienda/modelos/billetes1.pt')

        # CLASES:
        self.clsObject = ['', '', '', '', '', '', '', '', '', '',
                          '', '', '', '', '', '', '', '', '', '',
                          '', '', '', '', '', '', 'bolso', '', '',
                          '', '',
                          '', 'pelota_deportiva', '', '', '', '',
                          '',
                          '', 'botella', '', 'taza', 'tenedor', 'cuchillo', 'cuchara', '',
                          'platano',
                          'manzana', '', 'naranja', 'brocoli', 'zanahoria', 'hot_dog', '', 'donut', '',
                          '', '',
                          '', '', '', '', '', '', 'raton',
                          '',
                          'teclado', '', '', '', '', '', '',
                          '', '',
                          'jarron', 'tijeras', 'oso_de_peluche', '', 'cepillo_de_dientes']

        self.clsBillBank = ['Billete20', 'Billete100', 'Billete50', 'Billete10']

        self.total_balance = 0
        self.pay = ''

    # FUNCIONES DE DIBUJO
    def draw_area(self, img, color, xi, yi, xf, yf):
        img = cv2.rectangle(img, (xi, yi), (xf, yf), color, 1, 1)
        return img

    def draw_text(self, img, color, text, xi, yi, size, thickness, back=False):
        sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, size, thickness)
        dim = sizetext[0]
        baseline = sizetext[1]
        if back:
            img = cv2.rectangle(img, (xi, yi - dim[1] - baseline), (xi + dim[0], yi + baseline - 7), (0, 0, 0), cv2.FILLED)
        img = cv2.putText(img, text, (xi, yi - 5), cv2.FONT_HERSHEY_DUPLEX, size, color, thickness)
        return img

    def draw_line(self, img, color, xi, yi, xf, yf):
        img = cv2.line(img, (xi, yi), (xf, yf), color, 1, 1)
        return img

    def area(self, frame, xi, yi, xf, yf):
        al, an, c = frame.shape
        xi, yi = int(xi * an), int(yi * al)
        xf, yf = int(xf * an), int(yf * al)
        return xi, yi, xf, yf

    def marketplace_list(self, frame, object):
        list_products = {'bolso': 100, 'pelota_deportiva': 50, 'botella': 200, 'taza': 10, 'tenedor': 5, 'cuchillo': 8,
                         'cuchara': 5, 'platano': 3, 'manzana': 2, 'naranja': 2, 'brocoli': 2, 'zanahoria': 2,
                         'raton': 25, 'teclado': 120, 'hot_dog': 2, 'tijeras': 5, 'cepillo de dientes': 5, 'donut': 2,
                         'jarron': 10, 'oso_de_peluche': 30}

        list_area_xi, list_area_yi, list_area_xf, list_area_yf = self.area(frame, 0.7739, 0.6250, 0.9649, 0.9444)
        size_obj, thickness_obj = 0.60, 1

        if object not in [item[0] for item in self.shopping_list]:
            price = list_products[object]
            self.shopping_list.append([object, price])
            text = f'{object} --> ${price}'
            frame = self.draw_text(frame, (0, 255, 0), text, list_area_xi + 10,
                                   list_area_yi + (40 + (self.posicion_products * 20)),
                                   size_obj, thickness_obj, back=False)
            self.posicion_products += 1
            self.accumulative_price += price

        return frame

    def balance_process(self, bill_type):
        if bill_type == 'Billete10':
            self.balance = 10
        elif bill_type == 'Billete20':
            self.balance = 20
        elif bill_type == 'Billete50':
            self.balance = 50
        elif bill_type == 'Billete100':
            self.balance = 100

    def payment_process(self, accumulative_price, accumulative_balance):
        payment = accumulative_balance - accumulative_price
        if payment < 0:
            text = f'Falta cancelar {abs(payment)}$'
        elif payment > 0:
            text = f'Su cambio es de: {abs(payment)}$'
            self.accumulative_price = 0
            self.total_balance = 0
        else:
            text = f'Gracias por su compra!'
            self.accumulative_price = 0
            self.total_balance = 0

        return text

    def prediction_model(self, clean_frame, frame, model, clase):
        results = model(clean_frame, stream=True, verbose=False)
        for res in results:
            boxes = res.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1, x2, y2 = max(x1, 0), max(y1, 0), max(x2, 0), max(y2, 0)
                cls = int(box.cls[0])
                conf = math.ceil(box.conf[0])

                if clase == 0:
                    objeto = self.clsObject[cls]
                    text_obj = f'{objeto} {int(conf * 100)}%'
                    frame = self.marketplace_list(frame, objeto)
                    frame = self.draw_text(frame, (0, 255, 0), text_obj, x1, y1, 0.75, 1, back=True)
                    frame = self.draw_area(frame, (0, 255, 0), x1, y1, x2, y2)
                elif clase == 1:
                    bill_type = self.clsBillBank[cls]
                    text_obj = f'{bill_type} {int(conf * 100)}%'
                    self.balance_process(bill_type)
                    frame = self.draw_text(frame, (0, 255, 0), text_obj, x1, y1, 0.75, 1, back=True)
                    frame = self.draw_area(frame, (0, 255, 0), x1, y1, x2, y2)

        return frame

    def run(self):
        while True:
            ret, frame = self.cap.read()
            t = cv2.waitKey(5)

            clean_frame = frame.copy()
            self.shopping_list = []
            self.posicion_products = 1
            self.accumulative_price = 0
            self.balance = 0

            shop_area_xi, shop_area_yi, shop_area_xf, shop_area_yf = self.area(frame, 0.0351, 0.0486, 0.7539, 0.9444)
            frame = self.draw_area(frame, (0, 255, 0), shop_area_xi, shop_area_yi, shop_area_xf, shop_area_yf)
            frame = self.draw_text(frame, (0, 255, 0), 'Shopping area', shop_area_xi, shop_area_yf + 30, 0.75, 1)

            pay_area_xi, pay_area_yi, pay_area_xf, pay_area_yf = self.area(frame, 0.7739, 0.0486, 0.9649, 0.6050)
            frame = self.draw_line(frame, (0, 255, 0), pay_area_xi, pay_area_yi, pay_area_xi, int((pay_area_yi + pay_area_yf) / 2))
            frame = self.draw_line(frame, (0, 255, 0), pay_area_xi, pay_area_yi, int((pay_area_xi + pay_area_xf) / 2), pay_area_yi)
            frame = self.draw_line(frame, (0, 255, 0), pay_area_xf, int((pay_area_yi + pay_area_yf) / 2), pay_area_xf, pay_area_yf)
            frame = self.draw_line(frame, (0, 255, 0), int((pay_area_xi + pay_area_xf) / 2), pay_area_yf, pay_area_xf, pay_area_yf)
            frame = self.draw_text(frame, (0, 255, 0), 'Payment area', pay_area_xf - 100, shop_area_yi + 10, 0.50, 1)

            list_area_xi, list_area_yi, list_area_xf, list_area_yf = self.area(frame, 0.7739, 0.6250, 0.9649, 0.9444)
            frame = self.draw_line(frame, (0, 255, 0), list_area_xi, list_area_yi, list_area_xi, list_area_yf)
            frame = self.draw_line(frame, (0, 255, 0), list_area_xi, list_area_yi, list_area_xf, list_area_yi)
            frame = self.draw_line(frame, (0, 255, 0), list_area_xi + 30, list_area_yi + 30, list_area_xf - 30, list_area_yi + 30)
            frame = self.draw_text(frame, (0, 255, 0), 'Shopping List', list_area_xi + 55, list_area_yi + 30, 0.65, 1)

            frame = self.prediction_model(clean_frame, frame, self.ObjectModel, clase=0)
            frame = self.prediction_model(clean_frame, frame, self.billModel, clase=1)

            frame = self.draw_text(frame, (0, 255, 0), f'Compra total: {self.accumulative_price} $', list_area_xi + 10, list_area_yf, 0.60, 1)
            frame = self.draw_text(frame, (0, 255, 0), f'Saldo total: {self.total_balance} $', list_area_xi + 10, list_area_yf + 30, 0.60, 1)
            frame = self.draw_text(frame, (0, 255, 0), self.pay, list_area_xi - 300, list_area_yf + 30, 0.60, 1)

            cv2.imshow("Tienda IA", frame)

            if t == 83 or t == 115:
                self.total_balance += self.balance
                self.balance = 0

            if t == 80 or t == 112:
                self.pay = self.payment_process(self.accumulative_price, self.total_balance)

            if t == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tienda = ShopIA()
    tienda.run()
