### 使用 YOLO 和網路攝影機進行物體偵測
'''
辨別是在斑馬線上的行人或非斑馬線上的人，區別於摩托車與行人，first辨別斑馬線, second遍別在斑馬線上的人, third計算他們之間的距離, fourth 警示訊息
'''
import cv2
from ultralytics import YOLO # 使用ultralytics運算YOLO 
import numpy as np


## ultralytics
# conda activate conda
# conda install conda-forge::ultralytics
# pip install ultralytics

# conda install conda-forge::opencv
# conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
# https://docs.ultralytics.com/quickstart/#install-ultralytics

## 初始化影像捕獲 Initialize(初始化) video capture
# 透過攝影機捕獲影像；網路攝影機捕獲，建立一個VideoCapture物件並加它設定為預設攝影機(0)擷取影格，攝影機的分辨率設為640*480，到下"q"將退出
camera_capture = 0
# 透過影片捕獲影像；將視訊檔案作為辨識，不再是攝影機
video_capture = "/Users/wei/Files/Codes/Senior_Project/NeuralNetwork/YOLO/TestMaterials/行人0.mp4"

cap = cv2.VideoCapture(video_capture)
frame_width = int(cap.get(3))  # capture object width
frame_height = int(cap.get(4)) # capture object height
# print(frame_width, frame_height)

## adjust width and height based on myself needs https://boords.com/blog/what-is-16-9-aspect-ratio-definition-and-examples
resize_width = 1024
resize_height = 720 
# cap.set(3, 640)
# cap.set(4, 480)
if frame_width > 0: # 依照輸入捕獲物件的影像去調整大小，等比例縮放，並以高為720為標準去縮放寬度
    resize_height = int((resize_width / frame_width) * frame_height)

## 等待按鍵觸發waitKey的參數 使用圖片設定為"無限期地等待"，影片則是設定為"1ms等待"，兩者都是在按下q結束
key = 0
picture_exts = ['jpeg', 'jpg', 'bmp','png']
split_video_capture = video_capture.split('.')

key = 0 if split_video_capture[1] in picture_exts else 1

# Colors define
gree = (0, 255, 0) 
red = (0, 0, 255) 
white = (255, 255, 255) 
black = (0, 0, 0)


# load the yolo model
model = YOLO('yolov8n.pt') # https://docs.ultralytics.com/models/yolov8/

def DrawBoxesTexture(img, class_name, bounding_box, color):
    cv2.rectangle(img, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), color, 2)
    cv2.putText(img, f"{class_name}", (bounding_box[0], bounding_box[1]-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, white, 1)

def MergeBoundingBoxes(person, motorcycle):
    xmin1, ymin1, xmax1, ymax1 = person
    xmin2, ymin2, xmax2, ymax2 = motorcycle

    ## 檢查是否交集
    if (xmax1 < xmin2 or xmax2 < xmin1 or ymax1 < ymin2 or ymax2 < ymin1):
        return None
    else:
        x_min = min(xmin1, xmin2)
        y_min = min(ymin1, ymin2)
        x_max = max(xmax1, xmax2)
        y_max = max(ymax1, ymax2)
        return [x_min, y_min, x_max, y_max]


## 預測(推理)
def Predict(model, img, classes=[], conf=0.5, iou=0.5):
    img = cv2.resize(img, (resize_width, resize_height))
    if classes:
        results = model.predict(img, classes=classes, conf=conf, iou=0.5)
    else:
        results = model.predict(img, conf=conf, iou=0.5)
    # model: 使用的模型，YOLO模型經過訓練以偵測的物件類別清單；
    # img: img的Numpy到陣列，也可以直接是檔案的來源路徑
    # classes: 按照類別過濾結果，class=0或class=[0,2,3]
    # conf: 設定檢測目標的最小置信度閥值，若檢測到對象低於閥值則不予考慮
    return results

def PredictAndDetect(model, img, classes=[], conf=0.5, iou=0.5):
    img = cv2.resize(img, (resize_width, resize_height)) # 調整影像大小
    # 預測
    results = Predict(model, img, classes, conf=conf, iou=0.5)
    person_border = []
    motorcycle_border =  []
    a = 0
    for result in results:
        # result每單筆預測結果，results總預測結果
        for box in result.boxes:
            ## 查詢詳細結果（類別、座標、機率）
            class_name = result.names[int(box.cls[0])]
            coordinates = [round(x) for x in box.xyxy[0].tolist()]
            conf = round(box.conf[0].item(), 2)
            # print(f"Object type: {class_name}\nCoordinates: {Coordinates}\nProbability: {conf}\n---")
            
            if class_name == "person":
                person_border.append(coordinates)
            elif class_name == "motorcycle":
                motorcycle_border.append(coordinates)
            else: # others
                DrawBoxesTexture(img, class_name, coordinates, white)

        ## 合併機車與的機車上的人
        merged_borders = []
        for person_box in person_border:
            for motorcycle_box in motorcycle_border:
                merged_box =  MergeBoundingBoxes(person_box, motorcycle_box)
                if merged_box: 
                    DrawBoxesTexture(img, "motorcycle", merged_box, white)
                    merged_borders.append((person_box, motorcycle_box))
        
        # 過濾人與機車的bounding box是有交集的就將這組座標從
        #  person_border and motorcycle_border中移除
        for person_box, motorcycle_box in merged_borders: 
            if person_box in person_border:
                person_border.remove(person_box)
            if motorcycle_box in motorcycle_border:
                motorcycle_border.remove(motorcycle_box)
        
        # 過濾完後剩下的座標為沒有交集的部分
        for person_box in person_border:
            DrawBoxesTexture(img,"person", person_box, red)
        for motorcycle_box in motorcycle_border:
            DrawBoxesTexture(img,"motorcycle", motorcycle_box, white)
        

            '''
            # boxes 可用於索引、操作和將邊框轉換為不同格式；boxes.xyxy方框；boxes.cls方框的類別值；conf方框的值信度；
            # if result.names[int(box.cls[0])] == "person": # 只要是人就都用綠色線框起來
            #     ## 繪製方框
            #     cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])), (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 255, 0), 2)
            #     # cv2.rectangle(影像, 頂點座標, 對向頂點座標, 顏色, 線條寬度)
            #     # int(box.xyxy[0][0]) ~ int(box.xyxy[0][3]) = x_min, y_min, x_max, y_max
                
            #     ## 文字
            #     cv2.putText(img, f"{result.names[int(box.cls[0])]} {box.conf[0]:.2f}", (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
            #     # cv2.putText(影像, 文字(類別標籤名稱 及 置信度), 座標(x1, x1-10), 字型, 大小, 顏色, 線條寬度, 線條種類)
            #     # img的Numpy到陣列, 類別標籤名稱 置信度(可信度), (x1, x1-10), 字型, 大小為1, 綠色的 RGB Hex code, 1
            #     # print((int(box.xyxy[0][0]), int(box.xyxy[0][1])), (int(box.xyxy[0][2]), int(box.xyxy[0][3])))
            #     # print((img, f"{result.names[int(box.cls[0])]} {box.conf[0]:.2f}", (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1))
            
            # # elif result.names[int(box.cls[0])] == "motorcycle":
            #     motorcycle_border = (int(box.xyxy[0][0]), int(box.xyxy[0][1])), (int(box.xyxy[0][2]), int(box.xyxy[0][3]))
            
            # else:
            #     cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])), (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 0, 255), 2)
            #     cv2.putText(img, f"{result.names[int(box.cls[0])]} {box.conf[0]:.2f}", (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            #     
            #    # img = cv2.imread()
            #    # 是夠過該函數將圖片讀取進來，會將該資料儲存成一個Numpy到陣列，此陣列中的前兩個維度分別是圖片的高度與寬度，第三個維度是圖片的channel(通道；RGB彩色圖片的通道是3，灰階圖片則1)
            #    # 可以透過img.shape()查看陣列大小，假設(1080, 1920, 3)則表示高寬為 1080*1920彩色圖片
            ''' 

            
    return img, results


def main():
    skip_frames = 2 # 每3幀處理一次
    frame_count = 0
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    while True:
        success,img = cap.read() # success代表是否有讀取到影像/圖片；img表示讀取影像的像素成分
        if not success:
            break
        # 跳過frame以加快速度處理
        if frame_count % skip_frames != 0:
            continue
        img = cv2.resize(img, (resize_width, resize_height))

        # YOLO Model
        result_img, _ = PredictAndDetect(model, img, classes=[], conf=0.5, iou=0.5)

        # break # 只執行一次
        cv2.resizeWindow("Identify the results", resize_width, resize_height)
        cv2.imshow("Identify the results", result_img)
        if cv2.waitKey(key) == ord('q'):
            break
if __name__ == "__main__":
    main()

'''
## 虛擬碼
# 前期初始化
設定影像捕獲來源，看是要指定檔案還是攝影機
調整影像大小參數
設定輸出影像時的按鍵觸發參數
加載YOLO模型

# Predict And Detect
調整影像大小
進行模型預測
將每筆預測結果的類別、座標、機率記錄下來
如果預測結果是人或機車的話將座標放進list中，都不是的話就畫出邊界框與標記類別名稱
檢查每筆人與機車的座標是否有交集

# Merge Bounding Boxes
將座標拆解成他本該有的格式，格式為xmin, ymin, xmax, ymax
如果有交集的話，把x與y min取得人與機車的最小座標，x與y max取得人與機車最大的座標

# back to `PredictAndDetect後`
如果有交集就畫出邊界框與標記類別名稱，並將有交集的人與機車座標整合到一個合併座標的變數
如果人與機車的座標有在合併座標的變數當中，就把這組座標清除
將剩下的人與機車畫出邊界框與標記類別名稱
'''




# '''
# ###################################################################################################
# yolov8n.pt https://docs.ultralytics.com/models/yolov8/
# How to Open Camera in Linux? https://www.geeksforgeeks.org/how-to-open-camera-in-linux/
# # ultralytics
# conda install conda-forge::ultralytics 不行的話往下
# conda config --add channels conda-forge
# conda config --set channel_priority strict
# conda install ultralytics
# https://github.com/conda-forge/ultralytics-feedstock
# img = cv2.imread() https://blog.gtwang.org/programming/opencv-basic-image-read-and-write-tutorial/
# predict model https://docs.ultralytics.com/zh/modes/predict/#inference-sources
# boxes https://docs.ultralytics.com/zh/modes/predict/#boxes
# xyxy https://blog.csdn.net/dorisx/article/details/120253209
# rectangle putText https://blog.gtwang.org/programming/opencv-drawing-functions-tutorial/

# result.boxes.xyxy(ultralytics.engine.results.Boxes) https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Results.verbose
# yolov8計算兩物體的距離 https://docs.ultralytics.com/zh/guides/distance-calculation/
# 使用roboflow訓練YOLOv8自定義模型
# '''

# ###################################################################################################
# ## 模型預測 https://medium.com/@EricChou711/yolov8-介紹和手把手訓練自訂義模型-752d8d32cb73
# from ultralytics import YOLO

# # Load a model
# model = YOLO("yolov8n.pt")  # load an official model

# # Predict with the model
# results = model.predict(source="/Users/wei/Files/Codes/Senior_Project/NeuralNetwork/YOLO/TestMaterials/行人2.png",save=True)  # predict on an image

## fcn -> cnn -> rnn -> attention etc