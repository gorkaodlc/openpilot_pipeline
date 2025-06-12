/**
 * \defgroup  Lane_Detection rtmaps
 * lane_detection v0.1
 *
 * Copyright (C) 2017-2019, Vicomtech-IK4 (http://www.vicomtech.es/),
 * (Spain) all rights reserved.
 */


////////////////////////////////////////////////////////
// Integration on CarLOTA of the latest Lane Detection module in V4. It depends on Viulib feature_laneDetection
////////////

#ifndef _MAPSundistort_H
#define _MAPSundistort_H
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <cstdlib>
#include <opencv2/core/types_c.h>
#include <opencv2/videoio/videoio_c.h>
// Avoid redefining IplImage structure by RTMaps
#define __IPL_H__

// RTMaps SDK library
#include "maps.hpp"



class MAPSundistort: public MAPSComponent
{
  MAPS_COMPONENT_STANDARD_HEADER_CODE(MAPSundistort)

  public:
   /** DECLARE HERE PUBLIC FUNCTIONS IF NEEDED **/

  private:

   /** DECLARE HERE ANY OBJECT NEEDED DURING PROCESSING. EXAMPLE: THE LANE DETECTOR **/

   // Lane *lane;
  //  bool flip;
   /*!
    * \brief _first_frame  Flag indicating whether it is or not the first received frame. Used for initialization
    */
    bool _first_frame;
    bool _first_out_frame;
    std::string intrinsics_p;
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    cv::Mat map1, map2;
    int width,height;

   /** DECLARE HERE ANY PRIVATE METHODS NEEDED **/
};

#endif
