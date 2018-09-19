#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <opencv2/opencv.hpp>

#include "arcsoft_fsdk_face_detection.h"
#include "merror.h"

#define APPID     "DWnUNSz9cH9CbcYoEyFfjEXUCL7jZ7yCRa6aqN4BwKxb"
#define DETECT_SDKKEY    "J1FrMM5Gho9njpmF2YJgicXhBUN9kErWyAgU2T2czgpx"

#define INPUT_IMAGE_FORMAT  ASVL_PAF_I420
//#define INPUT_IMAGE_PATH    "your_input_image.yuv"
#define INPUT_IMAGE_WIDTH   (640)
#define INPUT_IMAGE_HEIGHT  (480)

#define WORKBUF_SIZE        (40*1024*1024)
#define MAX_FACE_NUM        (50)

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	MHandle hEngine = nullptr;
	MByte *pWorkMem = (MByte *)malloc(WORKBUF_SIZE);
	Mat img;
	if (pWorkMem == nullptr) {
		fprintf(stderr, "fail to malloc workbuf\r\n");
		exit(0);
	}

	int ret = AFD_FSDK_InitialFaceEngine(APPID, DETECT_SDKKEY, pWorkMem, WORKBUF_SIZE,
	                                     &hEngine, AFD_FSDK_OPF_0_HIGHER_EXT, 16, MAX_FACE_NUM);
	if (ret != 0) {
		fprintf(stderr, "fail to AFD_FSDK_InitialFaceEngine(): 0x%x\r\n", ret);
		free(pWorkMem);
		exit(0);
	}

	VideoCapture cap;
	cap.open("rtsp://10.166.129.58:8554/684387");

	while (1) {
		cap >> img;
		const AFD_FSDK_Version*pVersionInfo = AFD_FSDK_GetVersion(hEngine);

		ASVLOFFSCREEN inputImg = { 0 };
		inputImg.u32PixelArrayFormat = ASVL_PAF_RGB24_B8G8R8;
		inputImg.i32Width = img.cols;
		inputImg.i32Height = img.rows;
		inputImg.ppu8Plane[0] = img.data;
		if (!inputImg.ppu8Plane[0]) {
			fprintf(stderr, "fail to fu_ReadFile: %s\r\n", strerror(errno));
			AFD_FSDK_UninitialFaceEngine(hEngine);
			free(pWorkMem);
			exit(0);
		}

		inputImg.pi32Pitch[0] = inputImg.i32Width * 3;
		LPAFD_FSDK_FACERES faceResult;
		ret = AFD_FSDK_StillImageFaceDetection(hEngine, &inputImg, &faceResult);
		if (ret != 0) {
			fprintf(stderr, "fail to AFD_FSDK_StillImageFaceDetection(): 0x%x\r\n", ret);
			free(inputImg.ppu8Plane[0]);
			AFD_FSDK_UninitialFaceEngine(hEngine);
			free(pWorkMem);
			exit(0);
		}
		for (int i = 0; i < faceResult->nFace; i++) {
			printf("face %d:(%d,%d,%d,%d)\r\n", i,
			       faceResult->rcFace[i].left, faceResult->rcFace[i].top,
			       faceResult->rcFace[i].right, faceResult->rcFace[i].bottom);
			rectangle(img, cvPoint(faceResult->rcFace[i].left, faceResult->rcFace[i].top),
			          cvPoint(faceResult->rcFace[i].right, faceResult->rcFace[i].bottom), cvScalar(0, 0, 255), 3, 4, 0);
		}
		imshow("test", img);
		waitKey(1);
	}

	// free(inputImg.ppu8Plane[0]);
	AFD_FSDK_UninitialFaceEngine(hEngine);
	free(pWorkMem);

	return 0;
}
