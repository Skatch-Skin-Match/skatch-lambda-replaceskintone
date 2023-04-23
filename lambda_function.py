import cv2
import numpy as np
import requests
import boto3
import uuid
# from string import Template
import base64
from datetime import datetime


def lambda_handler(event, context): 
    now = datetime.now() 
    print("testtttt")
    print('event',event)
    url = ''
    userAvgHSV =  [0,0,0]
   
    print("111111111111111", now.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3])

    if(event):
        print("22222222222222222222", now.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3])

        if(event['queryStringParameters']):
           print('ebbbbbbbbbbbbbbbbb',event['queryStringParameters']['modelurl'])
           url = event['queryStringParameters']['modelurl']
           print("33333333333333", now.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3])
           userSkinVal = event['queryStringParameters']['userskintone']
           values = userSkinVal.split(",")
           userAvgHSV = [int(x) for x in values]

        else :
            url = 'https://d30ukgyabounne.cloudfront.net/hmgoepprod.jpg'
            userAvgHSV =  [0,0,0]


    else :
        url = 'https://d30ukgyabounne.cloudfront.net/hmgoepprod.jpg'
        
        # 'user.jpg'
        userAvgHSV =  [9.994970190385274, 106.37392379185503, 159.3702324879771]
        # 'fair'
        #userAvgHSV =  [20, 45.9, 239.7]
        # 'black'
        #userAvgHSV =  [20, 137.7, 112.2]
        


 
    print('final url',url)
    print("44444444444444", now.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3])


    response = requests.get(url)
    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)

    modelImg = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    # modelImg= cv2.imread('model.jpg')

    print("5555555555555555", now.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3])



    # Convert BGR to HSV
    modelHSV = cv2.cvtColor(modelImg, cv2.COLOR_BGR2HSV)





    # =-=-=-= UNOPTIMIZED MASKING BEGINS =-=-=-=

    # Define the lower and upper bounds of the skin tone in the HSV color space
    # (this is a blanket, unoptimized range. will be used to determine optimized range)
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])

    # Create a mask to segment the skin tone in the image
    modelMask = cv2.inRange(modelHSV, lower_skin, upper_skin)

    # Apply the mask to the original image to extract the skin tone area
    modelSkin = cv2.bitwise_and(modelImg, modelImg, mask=modelMask)
    modelSkinHSV = cv2.cvtColor(modelSkin, cv2.COLOR_BGR2HSV)

    # userAvgHSV = avgHSVCalc(userSkinHSV)
    # userAvgHSV =  [9.994970190385274, 106.37392379185503, 159.3702324879771]
    # userAvgHSV = [11.150035017960825, 100.31511646296003, 181.08731897973476]
    userAvgHSV = userAvgHSV


    print("666666666666666666", now.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]
)

    modelAvgHSV = avgHSVCalc(modelSkinHSV)
    
    print(f"Model Skin Avg HSV (UNOPTIMIZED): {modelAvgHSV}")
    print("77777777777777777", now.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]
)

    # =-=-=-= OPTIMIZED MASKING BEGINS =-=-=-=

    # Now that we have the HSV of the unoptimized model mask, we can optimize the mask.
    # If in LIGHT skin range
    if modelAvgHSV[2] >= 181:
        lower_skin = np.array([6, 20, 90])
        upper_skin = np.array([16, 255, 255])
        print ("LIGHT skin detected")

    # If in DARK skin range
    elif modelAvgHSV[2] < 181:
        lower_skin = np.array([0, 20, 45])
        upper_skin = np.array([14, 255, 255])
        print ("DARK skin detected")


    # Recalculating now that we have optimized HSV ranges.
    modelMask = cv2.inRange(modelHSV, lower_skin, upper_skin)
    modelSkin = cv2.bitwise_and(modelImg, modelImg, mask=modelMask)
    modelSkinHSV = cv2.cvtColor(modelSkin, cv2.COLOR_BGR2HSV)

    # userAvgHSV = avgHSVCalc(userSkinHSV)
    # userAvgHSV =  [9.994970190385274, 106.37392379185503, 159.3702324879771]
    # userAvgHSV = [11.150035017960825, 100.31511646296003, 181.08731897973476]
    userAvgHSV = userAvgHSV

    modelAvgHSV = avgHSVCalc(modelSkinHSV)
    
    print(f"Model Skin Avg HSV (OPTIMIZED): {modelAvgHSV}")
    print("78787878787878", now.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]
)

    # default adjustments
    reqAdjH = 0 # As of now, never shift hue.
    reqAdjS = 0
    reqAdjV = 0

    SCC = 112 # Saturation Cap Constant
    print("88888888888888", now.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3])


    if (userAvgHSV[1] < SCC and modelAvgHSV[1] < SCC):
        reqAdjS = userAvgHSV[1] - modelAvgHSV[1] # Neg if User is lighter, Pos if Model is lighter

    elif (userAvgHSV[1] >= SCC): # if user maxes out sat
        if (modelAvgHSV[1] < SCC): # if model is not max sat, max it out and match vib with user
            reqAdjS = SCC - modelAvgHSV[1] 
            reqAdjV = userAvgHSV[2] - modelAvgHSV[2]
        elif (modelAvgHSV[1] >= SCC): # else if model is already max sat, just match vib with user
            reqAdjV = userAvgHSV[2] - modelAvgHSV[2]

    elif (userAvgHSV[1] < SCC and modelAvgHSV[1] >= SCC):
        reqAdjS = userAvgHSV[1] - modelAvgHSV[1] # adj lowers Sat of model
        reqAdjV = userAvgHSV[2] - modelAvgHSV[2] # adj increases Vib of model

    print("HSV adjustments:")
    print(reqAdjH)
    print(reqAdjS)
    print(reqAdjV)
    print("999999999999999", now.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]
)

    modelResultHSV = modelSkinHSV.copy()
    # modelResultHSVImg = modelHSV.copy()

    # adjusting every pixel
    for i in modelResultHSV: # i is a particular row
        for n in i: # n is a partcular column
            if n[0]!=0 and n[1]!=0 and n[2]!=0: # skip black pixels
                # Enforcing floor/ceiling of 0/255
                
                # For Saturation
                if n[1] + reqAdjS > 255:
                    n[1] = 255
                elif n[1] + reqAdjS < 0:
                    n[1] = 0
                else:
                    n[1] = n[1] + reqAdjS

                # For Vibrance
                if n[2] + reqAdjV > 255:
                    n[2] = 255
                elif n[2] + reqAdjV < 0:
                    n[2] = 0
                else:
                    n[2] = n[2] + reqAdjV

     
    print("100000000000000000", now.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3])

    modelResultBGR = cv2.cvtColor(modelResultHSV, cv2.COLOR_HSV2BGR)
    non_skin_tone_mask = cv2.bitwise_not(modelMask)
    non_skin_tone = cv2.bitwise_and(modelImg, modelImg, mask=non_skin_tone_mask)

    blended_img = cv2.add(modelResultBGR, non_skin_tone)

    # cv2.imwrite('Input1.jpg',modelResultBGR)
    # cv2.imwrite('Input2.jpg',non_skin_tone_mask)
    # cv2.imwrite('Input3.jpg',non_skin_tone)
    # cv2.imwrite('Input4.jpg',blended_img)

    retval, buffer = cv2.imencode('.jpg', blended_img)
    modelResultB64 = base64.b64encode(buffer).decode('utf-8')
    print("11***********************", now.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]
)

    return {
        'headers': { "Content-Type": "image/png" },
        'statusCode': 200,
        'body': modelResultB64,
        'isBase64Encoded': True
    }

def avgHSVCalc(imgHSV):
    # Split HSV channels into separate arrays.
    imgHueChannel, imgSatChannel, imgVibChannel = cv2.split(imgHSV)

    # Getting average of each channels where pixel isn't black (0).
    imgHueChannel_np = np.array(imgHueChannel)
    imgSatChannel_np = np.array(imgSatChannel)
    imgVibChannel_np = np.array(imgVibChannel)

    avgImgHue = imgHueChannel_np.mean(where = imgHueChannel_np > 0)
    avgImgSat = imgSatChannel_np.mean(where = imgSatChannel_np > 0)
    avgImgVib = imgVibChannel_np.mean(where = imgVibChannel_np > 0)


    # HSV in CV2 is [179,255,255]. Conversions:
    # avgImgHue = (avgImgHue / 179) * 360
    # avgImgSat = (avgImgSat / 255) * 100
    # avgImgVib = (avgImgVib / 255) * 100
    # /\ If commented out, we are using 0-255 range for S&V.

    return [avgImgHue, avgImgSat, avgImgVib]


lambda_handler('','')


























