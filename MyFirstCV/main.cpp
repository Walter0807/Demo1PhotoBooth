#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <cstring>
#include <algorithm>
#include <queue>
#include <string>
#include <iostream>
#include <cstdio>
#include <map>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#define For(i,a,b) for(int i=a;i<=b;i++)

using namespace std;
using namespace cv;
string OutputPath;
string words;
int tot,CrownID,Pid,TieID,beardID;
Mat result;
Mat mask[8];
/** 函数声明 */
void detectAndDisplay( Mat frame);
void CrownIt(Mat frame,Point center, int face_width,int face_height)
{
    string CrownPath = "/Users/Walter/1/Demo1/resources/";
    char FileName[50];
    sprintf(FileName,"crown%d.jpg", CrownID % 6);
    CrownPath.append(FileName);
    Mat Crown = imread(CrownPath, 1);
    int Width = Crown.cols;
    int Height = Crown.rows;
    Mat Crown2;
    int heightcrown = cvRound(Height*face_width/(double)Width);
    resize(Crown,Crown2,Size(face_width,heightcrown),0,0,CV_INTER_LINEAR);
    Point starter(center.x - face_width*0.5, center.y - face_height*0.5-heightcrown);
    For(h,0,heightcrown-1)
    {
        //crown(i,j)->frame(starter.x+i, starter.y+j)
        if(h + starter.y<0 || h + starter.y>781) continue;
        Vec3b *q = frame.ptr<Vec3b>(h + starter.y);
        Vec3b *p = Crown2.ptr<Vec3b>(h);
        For(w,0,face_width-1)
        if (p[w][0]+p[w][1]+p[w][2]<711 && 0<starter.x + w&& starter.x + w<1282) q[starter.x + w] = p[w];
    }
}
void TieIt(Mat frame,Point center, int face_width,int face_height)
{
    string TiePath = "/Users/Walter/1/Demo1/resources/";
    char FileName[50];
    sprintf(FileName,"tie%d.jpg", TieID%4);
    TiePath.append(FileName);
    Mat Tie = imread(TiePath, 1);
    int Width = Tie.cols;
    int Height = Tie.rows;
    Mat Tie2;
    int heightTie = cvRound(Height*face_width/(double)Width);
    resize(Tie,Tie2,Size(face_width,heightTie),0,0,CV_INTER_LINEAR);
    Point starter(center.x - face_width*0.5, center.y + face_height*0.6);
    For(h,0,(heightTie)-1)
    {
        if(h + starter.y<0 || h + starter.y>781) continue;
        //crown(i,j)->frame(starter.x+i, starter.y+j)
        Vec3b *q = frame.ptr<Vec3b>(h + starter.y);
        Vec3b *p = Tie2.ptr<Vec3b>(h);
        For(w,0,(face_width)-1)
        if (p[w][0]+p[w][1]+p[w][2]<711 && 0<starter.x + w&& starter.x + w<1282) q[starter.x + w] = p[w];
    }
}



void Revert(Mat frame)
{
    int Width = frame.cols;
    int Height = frame.rows;
    For(i,0,Height-1)
    {
        Vec3b *cur = frame.ptr<Vec3b>(i);
        For(j,0,(Width-1)/2)
        {
            Vec3b tmp = cur[j];
            cur[j] = cur[Width-1-j];
            cur[Width-1-j] = tmp;
        }
        
    }
}
void TakePhoto(Mat frame)
{
    tot++;
    OutputPath = "/Users/Walter/1/Demo1/Shots/";
    char FileName[50];
    sprintf(FileName,"Photo%d.jpg", tot);
    OutputPath.append(FileName);
    cout<<OutputPath<<" Saved"<<endl;
    imwrite(OutputPath,frame);
    
}

/** 全局变量 */
string face_cascade_name = "/Users/Walter/1/Demo1/R/haarcascade_frontalface_alt.xml";
string eyes_cascade_name = "/Users/Walter/1/Demo1/R/haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);

int main( int argc, const char** argv )
{
    Mat frame;
    //-- 1. 加载级联分类器文件
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    //-- 2. 打开内置摄像头视频流
    CvCapture* capture = cvCreateCameraCapture(CV_CAP_ANY);  
    if( capture )
    {
        while( true )
        {           
            frame = cvQueryFrame( capture );          
            //-- 3. 对当前帧使用分类器进行检测
            if( !frame.empty() )
            {     Revert(frame); detectAndDisplay( frame ); }
            else
            { printf(" --(!) No captured frame -- Break!"); break; }
            
            int c = waitKey(10);
            if(c == 27) { break; }
            if((char)c == 's')            TakePhoto(result);
            if((char)c == 'x')            CrownID++;
            if((char)c == 'w')            Pid++;
            if((char)c == 'c')            TieID++;

        }
    }
    return 0;
}
void Sketch(Mat frame)
{
    int width=frame.cols;
    int height=frame.rows;
    //去色
    Mat gray0, gray1;
    cvtColor(frame,gray0,CV_BGR2GRAY);
    //反色
    addWeighted(gray0,-1,NULL,0,255,gray1);
    //高斯模糊,高斯核的Size与最后的效果有关
    GaussianBlur(gray1,gray1,Size(11,11),0);
    
    //融合：颜色减淡
    Mat tmp(gray1.size(),CV_8UC1);
    For(y,0,height-1)
    {
        
        uchar* P0 = gray0.ptr<uchar>(y);
        uchar* P1 = gray1.ptr<uchar>(y);
        uchar* P  = tmp.ptr<uchar>(y);
        For(x,0,width-1)
        {
            int tmp0=P0[x];
            int tmp1=P1[x];
            P[x] =(uchar) min((tmp0+(tmp0*tmp1)/(256-tmp1)),255);
        }
        
    }
    result = tmp;
}

void Feather(Mat frame)
{
    float mSize = 0.5;
    int width=frame.cols;
    int heigh=frame.rows;
    int centerX=width>>1;
    int centerY=heigh>>1;
    
    int maxV=centerX*centerX+centerY*centerY;
    int minV=(int)(maxV*(1-mSize));
    int diff= maxV -minV;
    float ratio = width >heigh ? (float)heigh/(float)width : (float)width/(float)heigh;
    
    Mat dst(frame.size(),CV_8UC3);
    for (int y=0;y<heigh;y++)
    {
        uchar* frameP=frame.ptr<uchar>(y);
        uchar* dstP=dst.ptr<uchar>(y);
        for (int x=0;x<width;x++)
        {
            int b=frameP[3*x];
            int g=frameP[3*x+1];
            int r=frameP[3*x+2];
            float dx=centerX-x;
            float dy=centerY-y;
            if(width > heigh)
                dx= (dx*ratio);
            else
                dy = (dy*ratio);
            int dstSq = dx*dx + dy*dy;
            float v = ((float) dstSq / diff)*255;
            r = (int)(r +v);
            g = (int)(g +v);
            b = (int)(b +v);
            r = (r>255 ? 255 : (r<0? 0 : r));
            g = (g>255 ? 255 : (g<0? 0 : g));
            b = (b>255 ? 255 : (b<0? 0 : b));
            dstP[3*x] = (uchar)b;
            dstP[3*x+1] = (uchar)g;
            dstP[3*x+2] = (uchar)r;
        }
    }
    result = dst;
}

void Retro(Mat frame)
{
    int width=frame.cols;
    int heigh=frame.rows;
    for (int y=0;y<heigh;y++)
    {
        uchar* frameP=frame.ptr<uchar>(y);
        uchar* resultP=result.ptr<uchar>(y);
        for (int x=0;x<width;x++)
        {
            float b=frameP[3*x];
            float g=frameP[3*x+1];
            float r=frameP[3*x+2];
            float newB=0.272*r+0.534*g+0.131*b;
            float newG=0.349*r+0.686*g+0.168*b;
            float newR=0.393*r+0.769*g+0.189*b;
            newB = (newB>255 ? 255 : (newB<0? 0 : newB));
            newG = (newG>255 ? 255 : (newG<0? 0 : newG));
            newR = (newR>255 ? 255 : (newR<0? 0 : newR));
            resultP[3*x] = (uchar)newB;
            resultP[3*x+1] = (uchar)newG;
            resultP[3*x+2] = (uchar)newR;
        }
    }
}
void woodcut(Mat frame)
{
    Mat dstImage = frame;
    cvtColor(dstImage,dstImage,CV_BGR2GRAY);
    threshold(dstImage,dstImage,127,255,THRESH_BINARY);
    result = dstImage;
}

void Process(Mat frame, int i)
{
    if(i==0)
    {result = frame;
     words = "Original";
    }
    if(i==1) {filter2D(frame,result,frame.depth(),mask[i]); words = "Sharp";}
    if(i==2) {filter2D(frame,result,frame.depth(),mask[i]); words = "Blur";}
    if(i==3) {Sketch(frame);words = "Sketch";}
    if(i==4) {Feather(frame);words = "Feather";}
    if(i==5) {Retro(frame);words = "Retro";}
    if(i==6) {woodcut(frame);words = "Woodcut";}

}
/** @函数 detectAndDisplay */
void detectAndDisplay( Mat frame )
{
    std::vector<Rect> faces;
    Mat frame_gray;
    
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
     
    //-- 多尺寸检测人脸
    face_cascade.detectMultiScale( frame_gray, faces, 1.2, 3, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    
    for( int i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
     //   ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

        Mat faceROI = frame_gray( faces[i] );
        std::vector<Rect> eyes;
        
        //-- 在每张人脸上检测双眼
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );
        if (eyes.size()==0) continue;//双重检测去除误判
        CrownIt(frame, center, faces[i].width, faces[i].height);
        TieIt(frame,center,faces[i].width, faces[i].height);
        for( int j = 0; j < eyes.size(); j++ )
        {
            Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
            int radius = cvRound( (eyes[j].width + eyes[i].height)*0.25 );
          //  circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
        }
    }
    //-- 显示结果图像
    mask[1] =(Mat_<double>(3,3) <<  0,-1.25,0,
    							   -1.25,7,-1.25,
    							   0,-1.25,0);
    mask[2] = Mat::ones(3,3, CV_32F);
    mask[2]/=7;
    Process(frame, Pid%7);
    words = "Mode:"+words;
    putText( result, words, Point( 80,80),CV_FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar(230, 230, 230));
    imshow( window_name, result );
}
