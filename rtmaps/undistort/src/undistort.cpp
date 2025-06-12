/**
 * \defgroup  opencv undistort function for rtmaps
 * lane_detection v0.1
 *
 * Copyright (C) 2017-2019, Vicomtech-IK4 (http://www.vicomtech.es/),
 * (Spain) all rights reserved.
 */


////////////////////////////////////////////////////////
////////////

#include "undistort.h"
using namespace cv;
// Input image
MAPS_BEGIN_INPUTS_DEFINITION(MAPSundistort)
   MAPS_INPUT("imageIn",MAPS::FilterIplImage,MAPS::FifoReader)
MAPS_END_INPUTS_DEFINITION

// Output image
MAPS_BEGIN_OUTPUTS_DEFINITION(MAPSundistort)
  MAPS_OUTPUT("imageOut", MAPS::IplImage, NULL, NULL, 0)
MAPS_END_OUTPUTS_DEFINITION

// Properties that user needs and/or can configure
MAPS_BEGIN_PROPERTIES_DEFINITION(MAPSundistort)
  // Examples:
  //MAPS_PROPERTY_ENUM("framework","Caffe|Tensorflow|Torch",0,false,false)
  MAPS_PROPERTY_SUBTYPE("Intrinsics_File", (char*)NULL, true, false, MAPS::PropertySubTypeFile | MAPS::PropertySubTypeMustExist)

  
MAPS_END_PROPERTIES_DEFINITION

// Add here more actions to your components
MAPS_BEGIN_ACTIONS_DEFINITION(MAPSundistort)
MAPS_END_ACTIONS_DEFINITION

// Define your ImageProcessing name, version, behaviour, etc. Define also its inputs, outputs, properties, and actions number.
MAPS_COMPONENT_DEFINITION(MAPSundistort,"undistort","1.0",128,MAPS::Sequential|MAPS::Threaded,
			  MAPS::Threaded,
			  -1, // Nb of inputs. Leave -1 to use the number of declared input definitions
			  -1, // Nb of outputs. Leave -1 to use the number of declared output definitions
			  -1, // Nb of properties. Leave -1 to use the number of declared property definitions
			  -1) // Nb of actions. Leave -1 to use the number of declared action definition


/*!
 * \brief MAPSundistort::Birth
 * Component birth is executed only once at RTMaps module startup.
 * Initialization of variables. Read input properties
 */
void MAPSundistort::Birth()
{
  /******* EXAMPLES for reading properties *********/
  /*int framework = GetIntegerProperty("framework");
  if (framework>0){
    ReportWarning("The current version only supports Caffe framework", 1);
    Die();
    Wait4Death();
  }
*/
  MAPSString strMaps = GetStringProperty("Intrinsics_File");
  MAPSIconv::localeChar *localef = MAPSIconv::UTF8ToLocale(strMaps);
  intrinsics_p = localef;
  MAPSIconv::releaseLocale(localef);




    FileStorage fs;
    fs.open(intrinsics_p, FileStorage::READ);
    if( !fs.isOpened() ){
        std::cerr << " Fail to open " << intrinsics_p << std::endl;
        exit(EXIT_FAILURE);
    }

    // Get camera parameters
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs; 
    fs["image_width"] >> width;
    fs["image_height"]>> height;

    // Print out the camera parameters
    std::cout << "\n -- Camera parameters -- " << std::endl;
    std::cout << "\n CameraMatrix = " << std::endl << " " << cameraMatrix << std::endl << std::endl;
    std::cout << " Distortion coefficients = " << std::endl << " " << distCoeffs << std::endl << std::endl;

    fs.release();



_first_frame = true;
  _first_out_frame = true;
}


/*!
 * \brief MAPSundistort::Core
 * The Core() method. This is executed in a loop while RTMaps diagram is running
 */
void MAPSundistort::Core()
{
  // Prepare data containers
  cv::Mat iimg, oimg,grayimg, outputImg;
  //std::cout << "\n -- Core-- " << std::endl;
  // Request input image data
  // This function will block until a new data is available on the input.
  MAPSIOElt* ioEltIn = StartReading(Input("imageIn"));
  if (NULL==ioEltIn)
    return;
 

  IplImage& imageIn = ioEltIn->IplImage();

    //printf("for loop\n");
    iimg= cv::cvarrToMat(&imageIn);


   //undistort(iimg,oimg,cameraMatrix,distCoeffs);
if (_first_out_frame) {
   cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), cameraMatrix, cv::Size(width,height), CV_32FC1, map1, map2);

}

cv::remap(iimg, oimg, map1, map2, cv::INTER_LINEAR);


  // Allocate output buffer for the image
  if (_first_out_frame) {
    // This allocation does not transfer ROI details correctly. Do an image clone first
    cv::Mat auxoimg = oimg.clone();
    IplImage ipltemp = MAPS::IplImageModel(auxoimg.cols, auxoimg.rows, MAPS_CHANNELSEQ_BGR, IPL_DATA_ORDER_PIXEL, IPL_DEPTH_8U);
    Output("imageOut").AllocOutputBufferIplImage(ipltemp);
    _first_out_frame = false;
  }

  //Output buffers are allocated at this point. We can request a MAPSIOElt object for the
  //output buffers to write in.
  MAPSIOElt* ioEltOut = StartWriting(Output("imageOut"));
  IplImage& imageOut = ioEltOut->IplImage();
  cv::Mat tmp=cv::cvarrToMat(&imageOut);//(imageOut.height, imageOut.width, CV_8UC3, imageOut.imageData);
  //cv::cvtColor(oimg, oimg, cv::COLOR_BGR2RGB);
  oimg.copyTo(tmp);



  //outputImg.copyTo();
  //iimg.copyTo(tmp);

  //Transfer the timestamp
  ioEltOut->Timestamp() = ioEltIn->Timestamp();

  // Send the output buffer (and don't write into it any more !)
  StopWriting(ioEltOut);

}


/*!
 * \brief MAPSundistort::Death()
 *  The Death() method is executed when RTMaps diagram stops running and the component is stopped
 */
void MAPSundistort::Death()
{

//delete(lane);
}
