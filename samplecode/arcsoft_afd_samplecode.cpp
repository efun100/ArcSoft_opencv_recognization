#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <opencv2/opencv.hpp>

#include "arcsoft_fsdk_face_detection.h"
#include "arcsoft_fsdk_face_recognition.h"
#include "merror.h"

#define APPID     "DWnUNSz9cH9CbcYoEyFfjEXUCL7jZ7yCRa6aqN4BwKxb"
#define DETECT_SDKKEY    "J1FrMM5Gho9njpmF2YJgicXhBUN9kErWyAgU2T2czgpx"
#define RECOGNIZE_SDKKEY    "J1FrMM5Gho9njpmF2YJgicYBq5Qnzhdhas463YYr6DWS"

#define WORKBUF_SIZE        (40*1024*1024)
#define MAX_FACE_NUM        (50)

using namespace std;
using namespace cv;

AFR_FSDK_FACEMODEL faceModels1 = { 0 };
MHandle hRecognizeEngine = nullptr;
MHandle hDetectEngine = nullptr;

int init_models()
{
	Mat model;
	ASVLOFFSCREEN modelImg = { 0 };
	model = imread("model.jpg");
	// model = imread("liuyifei.jpg");
	cout << model.cols << endl;
	cout << model.rows << endl;

	modelImg.u32PixelArrayFormat = ASVL_PAF_RGB24_B8G8R8;
	modelImg.i32Width = model.cols;
	modelImg.i32Height = model.rows;
	modelImg.ppu8Plane[0] = model.data;
	if (!modelImg.ppu8Plane[0]) {
		fprintf(stderr, "fail to fu_ReadFile: %s\r\n", strerror(errno));
		exit(0);
	}

	modelImg.pi32Pitch[0] = modelImg.i32Width * 3;

	AFR_FSDK_FACEINPUT faceResult;
	faceResult.lOrient = AFR_FSDK_FOC_0;
	faceResult.rcFace.left = 74;
	faceResult.rcFace.top = 196;
	faceResult.rcFace.right = 370;
	faceResult.rcFace.bottom = 492;
	AFR_FSDK_FACEMODEL LocalFaceModels = { 0 };
	int ret = AFR_FSDK_ExtractFRFeature(hRecognizeEngine, &modelImg, &faceResult, &LocalFaceModels);
	if (ret != 0) {
		fprintf(stderr, "fail to AFR_FSDK_ExtractFRFeature in Image A\r\n");
		exit(0);
	}
	faceModels1.lFeatureSize = LocalFaceModels.lFeatureSize;
	faceModels1.pbFeature = (MByte*)malloc(faceModels1.lFeatureSize);
	memcpy(faceModels1.pbFeature, LocalFaceModels.pbFeature, faceModels1.lFeatureSize);
}

int main(int argc, char* argv[])
{
	MByte *pDetectWorkMem = (MByte *)malloc(WORKBUF_SIZE);
	Mat img;
	if (pDetectWorkMem == nullptr) {
		fprintf(stderr, "fail to malloc workbuf\r\n");
		exit(0);
	}

	int ret = AFD_FSDK_InitialFaceEngine(APPID, DETECT_SDKKEY, pDetectWorkMem, WORKBUF_SIZE,
	                                     &hDetectEngine, AFD_FSDK_OPF_0_HIGHER_EXT, 16, MAX_FACE_NUM);
	if (ret != 0) {
		fprintf(stderr, "fail to AFD_FSDK_InitialFaceEngine(): 0x%x\r\n", ret);
		free(pDetectWorkMem);
		exit(0);
	}

	MByte *pRecognizeWorkMem = (MByte *)malloc(WORKBUF_SIZE);
	if (pRecognizeWorkMem == nullptr) {
		fprintf(stderr, "fail to malloc engine work buffer\r\n");
		exit(0);
	}

	ret = AFR_FSDK_InitialEngine(APPID, RECOGNIZE_SDKKEY, pRecognizeWorkMem, WORKBUF_SIZE, &hRecognizeEngine);
	if (ret != 0) {
		fprintf(stderr, "fail to AFR_FSDK_InitialEngine(): 0x%x\r\n", ret);
		free(pRecognizeWorkMem);
		exit(0);
	}

	init_models();
	VideoCapture cap;
	cap.open("rtsp://10.166.129.58:8554/684387");

	while (1) {
		cap >> img;
		// img = imread("liuyifei.jpg");
		const AFD_FSDK_Version*pVersionInfo = AFD_FSDK_GetVersion(hDetectEngine);

		ASVLOFFSCREEN inputImg = { 0 };
		inputImg.u32PixelArrayFormat = ASVL_PAF_RGB24_B8G8R8;
		inputImg.i32Width = img.cols;
		inputImg.i32Height = img.rows;
		inputImg.ppu8Plane[0] = img.data;
		if (!inputImg.ppu8Plane[0]) {
			fprintf(stderr, "fail to fu_ReadFile: %s\r\n", strerror(errno));
			AFD_FSDK_UninitialFaceEngine(hDetectEngine);
			free(pDetectWorkMem);
			exit(0);
		}

		inputImg.pi32Pitch[0] = inputImg.i32Width * 3;
		LPAFD_FSDK_FACERES faceResults;
		ret = AFD_FSDK_StillImageFaceDetection(hDetectEngine, &inputImg, &faceResults);
		if (ret != 0) {
			fprintf(stderr, "fail to AFD_FSDK_StillImageFaceDetection(): 0x%x\r\n", ret);
			free(inputImg.ppu8Plane[0]);
			AFD_FSDK_UninitialFaceEngine(hDetectEngine);
			free(pDetectWorkMem);
			exit(0);
		}
		for (int i = 0; i < faceResults->nFace; i++) {
			printf("face %d:(%d,%d,%d,%d)\r\n", i,
			       faceResults->rcFace[i].left, faceResults->rcFace[i].top,
			       faceResults->rcFace[i].right, faceResults->rcFace[i].bottom);
			// int height = faceResults->rcFace[i].bottom - faceResults->rcFace[i].top;
			// int width = faceResults->rcFace[i].right - faceResults->rcFace[i].left;
			// Mat image_roi = img(Rect(faceResults->rcFace[i].left, faceResults->rcFace[i].top, width, height));
			// imwrite("model.jpg", image_roi);
			// imshow("liuyifei", image_roi);
			// waitKey(0);

			ASVLOFFSCREEN faceImg = { 0 };
			faceImg.u32PixelArrayFormat = ASVL_PAF_RGB24_B8G8R8;
			faceImg.i32Width = img.cols;
			faceImg.i32Height = img.rows;
			faceImg.ppu8Plane[0] = img.data;
			if (!faceImg.ppu8Plane[0]) {
				fprintf(stderr, "fail to fu_ReadFile: %s\r\n", strerror(errno));
				exit(0);
			}

			faceImg.pi32Pitch[0] = faceImg.i32Width * 3;

			AFR_FSDK_FACEMODEL faceModels2 = { 0 };
			AFR_FSDK_FACEMODEL LocalFaceModels = { 0 };
			AFR_FSDK_FACEINPUT faceResult;
			faceResult.lOrient = AFR_FSDK_FOC_0;
			faceResult.rcFace.left = faceResults->rcFace[i].left;
			faceResult.rcFace.top = faceResults->rcFace[i].top;
			faceResult.rcFace.right = faceResults->rcFace[i].right;
			faceResult.rcFace.bottom = faceResults->rcFace[i].bottom;
			ret = AFR_FSDK_ExtractFRFeature(hRecognizeEngine, &faceImg, &faceResult, &LocalFaceModels);
			if (ret != 0) {
				fprintf(stderr, "fail to AFR_FSDK_ExtractFRFeature in Image B\r\n");
				continue;
			}
			// imwrite("model.jpg", img);
			// goto end;

			faceModels2.lFeatureSize = LocalFaceModels.lFeatureSize;
			faceModels2.pbFeature = (MByte*)malloc(faceModels2.lFeatureSize);
			memcpy(faceModels2.pbFeature, LocalFaceModels.pbFeature, faceModels2.lFeatureSize);

			MFloat fSimilScore = 0.0f;
			ret = AFR_FSDK_FacePairMatching(hRecognizeEngine, &faceModels1, &faceModels2, &fSimilScore);
			printf("fSimilScore ==  %f\r\n", fSimilScore);
			char text[100] = {0};
			if (fSimilScore > 0.75) {
				sprintf(text, "zhuqighua %f", fSimilScore);
				putText(img, text, Point(faceResults->rcFace[i].left, faceResults->rcFace[i].top),
				        FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2);
			} else {
				putText(img, "unknown", Point(faceResults->rcFace[i].left, faceResults->rcFace[i].top),
				        FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2);
			}
			rectangle(img, cvPoint(faceResults->rcFace[i].left, faceResults->rcFace[i].top),
			          cvPoint(faceResults->rcFace[i].right, faceResults->rcFace[i].bottom), cvScalar(0, 0, 255), 3, 4, 0);
		}
		imshow("test", img);
		waitKey(1);
	}

end:
	// free(inputImg.ppu8Plane[0]);
	AFD_FSDK_UninitialFaceEngine(hDetectEngine);
	free(pDetectWorkMem);

	return 0;
}
