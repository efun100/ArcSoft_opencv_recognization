#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>


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

MHandle hRecognizeEngine = nullptr;
MHandle hDetectEngine = nullptr;

struct modelInfo {
	char name[100];
	vector<AFR_FSDK_FACEMODEL> faceModels;
};

AFR_FSDK_FACEMODEL getAFR_FSDK_FACEMODELFromMat(Mat img)
{
	AFR_FSDK_FACEMODEL faceModel = { 0 };
	ASVLOFFSCREEN modelImg = { 0 };

	// model = imread("liuyifei.jpg");
	cout << img.cols << endl;
	cout << img.rows << endl;

	modelImg.u32PixelArrayFormat = ASVL_PAF_RGB24_B8G8R8;
	modelImg.i32Width = img.cols;
	modelImg.i32Height = img.rows;
	modelImg.ppu8Plane[0] = img.data;
	if (!modelImg.ppu8Plane[0]) {
		fprintf(stderr, "fail to fu_ReadFile: %s\r\n", strerror(errno));
		exit(0);
	}

	modelImg.pi32Pitch[0] = modelImg.i32Width * 3;

	LPAFD_FSDK_FACERES faceResults;
	int ret = AFD_FSDK_StillImageFaceDetection(hDetectEngine, &modelImg, &faceResults);
	if (ret != 0) {
		fprintf(stderr, "fail to AFD_FSDK_StillImageFaceDetection(): 0x%x\r\n", ret);
		exit(0);
	}

	if (faceResults->nFace != 1) {
		cout << "faceResult != 1" << endl;
		exit(0);
	}

	AFR_FSDK_FACEINPUT faceResult;
	faceResult.lOrient = AFR_FSDK_FOC_0;
	faceResult.rcFace.left = faceResults->rcFace[0].left;
	faceResult.rcFace.top = faceResults->rcFace[0].top;
	faceResult.rcFace.right = faceResults->rcFace[0].right;
	faceResult.rcFace.bottom = faceResults->rcFace[0].bottom;
	AFR_FSDK_FACEMODEL LocalFaceModels = { 0 };
	ret = AFR_FSDK_ExtractFRFeature(hRecognizeEngine, &modelImg, &faceResult, &LocalFaceModels);
	if (ret != 0) {
		fprintf(stderr, "fail to AFR_FSDK_ExtractFRFeature in Image A\r\n");
		exit(0);
	}
	faceModel.lFeatureSize = LocalFaceModels.lFeatureSize;
	faceModel.pbFeature = (MByte*)malloc(faceModel.lFeatureSize);
	memcpy(faceModel.pbFeature, LocalFaceModels.pbFeature, faceModel.lFeatureSize);
	return faceModel;
}

vector<modelInfo> modelInfos;

void init_models()
{
	// Mat model;
	// model = imread("model.jpg");
	// faceModels1 = getAFR_FSDK_FACEMODELFromMat(model);

	DIR *dfd;
	char *pathname = "../models";
	char name[MAX_PATH];
	struct dirent *dp;
	if ((dfd = opendir(pathname)) == NULL) {
		printf("dir_order: can't open %s\n %s", pathname, strerror(errno));
		return;
	}

	while ((dp = readdir(dfd)) != NULL) {
		if (strncmp(dp->d_name, ".", 1) == 0)
			continue; /* 跳过当前文件夹和上一层文件夹以及隐藏文件*/

		if (strlen(pathname) + strlen(dp->d_name) + 2 > sizeof(name)) {
			printf("dir_order: name %s %s too long\n", pathname, dp->d_name);
		} else {
			struct stat s_buf;
			memset(name, 0, sizeof(name));
			sprintf(name, "%s/%s", pathname, dp->d_name);
			cout << name << endl;
			stat(name, &s_buf);
			if (S_ISDIR(s_buf.st_mode)) {
				DIR *subDfd = opendir(name);
				struct dirent *subdp;
				struct modelInfo m;
				strcpy(m.name, dp->d_name);
				while ((subdp = readdir(subDfd)) != NULL) {
					cout << subdp->d_name << endl;
					if (strncmp(subdp->d_name, ".", 1) == 0)
						continue; /* 跳过当前文件夹和上一层文件夹以及隐藏文件*/
					if (!strstr(subdp->d_name, ".jpg"))
						continue;

					char imgname[MAX_PATH];
					memset(imgname, 0 , sizeof(imgname));
					if (strlen(name) + strlen(subdp->d_name) + 2 > sizeof(imgname)) {
						printf("%s: name %s %s too long\n", __func__, name, dp->d_name);
					} else {
						sprintf(imgname, "%s/%s", name, subdp->d_name);
						cout << imgname << endl;
						Mat img = imread(imgname);
						if (!img.data) {
							continue;
						}
						m.faceModels.push_back(getAFR_FSDK_FACEMODELFromMat(img));
					}
				}
				modelInfos.push_back(m);
				closedir(subDfd);
			}
		}
	}
	closedir(dfd);
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

	cout << __LINE__ << endl;
	for (auto it = modelInfos.begin(); it != modelInfos.end(); ++it)
		cout << it->name << endl;

	VideoCapture cap;
	cap.open("rtsp://10.230.7.94:8554/684387");

	while (1) {
		cap >> img;

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
			int similarIndex = 0;
			float similarScore = 0.0;
			for (int j = 0; j < modelInfos.size(); j++) {
				for (int k = 0; k < modelInfos[j].faceModels.size(); k++) {
					ret = AFR_FSDK_FacePairMatching(hRecognizeEngine, &modelInfos[j].faceModels[k], &faceModels2, &fSimilScore);
					if (fSimilScore > 0.7 && fSimilScore > similarScore) {
						similarScore = fSimilScore;
						similarIndex = j;
					}
				}
			}
			printf("fSimilScore == %f\r\n", fSimilScore);
			char text[100] = {0};
			if (similarScore > 0.75) {
				sprintf(text, "%s %f", modelInfos[similarIndex].name, similarScore);
				putText(img, text, Point(faceResults->rcFace[i].left, faceResults->rcFace[i].top),
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
