/*******************************************************************************
*   Based on code under copyrights below, supplied as part of the Chilitags    *
*   project: https://github.com/chili-epfl/chilitags                           *
*   Copyright 2013-2014 EPFL                                                   *
*   Copyright 2013-2014 Quentin Bonnard                                        *
*   Additions made by David Swarbrick 2020                                     *
*******************************************************************************/

#include <chilitags.hpp>

#include <opencv2/core/utility.hpp> // getTickCount...
#include <opencv2/imgproc/imgproc.hpp>


#include <opencv2/core/core_c.h> // CV_AA

#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <fstream>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <cmath>

static const float CALIBRATION_X = 1.0f;
static const float CALIBRATION_Y = 1.0f;

void drawTags(
    cv::Mat outputImage,
    const chilitags::TagCornerMap &tags,
    int64 startTime,
    int64 endTime,
    bool detection
    );

void logTagPosition(
  cv::Mat outputImage,
  const chilitags::TagCornerMap &tags,
  int64 startTime,
  int64 endTime
);

//   RelativeChilitags();
 // ~RelativeChilitags ();
class RelativeChilitags {
private:
  chilitags::TagCornerMap tags;
  float xConv;
  float yConv;
  float rotation;
  bool mapUpdated = false;
  std::ofstream logFile;

public:
  RelativeChilitags (void);
  ~RelativeChilitags(void);
  void        updateCornerMap(chilitags::TagCornerMap &newTags);
  cv::Point2f centreOfTag(chilitags::Quad tagRep);
  void        calcCalibrationFactors(void);
  cv::Point2f offsetAndRotate(cv::Point2f position);
  cv::Point2f scaleOffsetAndRotate(cv::Point2f position);
  float       tagRelativeOrientation(chilitags::Quad tag);
  cv::Point2f tagRelativePosition(chilitags::Quad tag);
  void        logNovelTagLocations(void);
};

RelativeChilitags::RelativeChilitags () {
  // cv::Matx<float, 4, 2> q(0.0, 0.0,0.0, 0.0,0.0, 0.0, 0.0, 0.0);
  // chilitags::TagCornerMap m({0,q});
  // tags = m;

  // Initialize private tag variable
  chilitags::Quad q(0.0, 0.0,0.0, 0.0,0.0, 0.0, 0.0, 0.0);
  std::map<int, chilitags::Quad> tags = {
    std::pair<int, chilitags::Quad> (0,q),
  };
  // _xRes = xRes;
  // _yRes = yRes;
  xConv = 1.0f;
  yConv = 1.0f;
  rotation = 0.0f;

  // Set up logging file, noting that Turtlebot logging uses this format for naming:
  // "Turtlebot_position_log-{}-{}-{}-{}:{}.csv".format(
  // dt.year, dt.month, dt.day, dt.hour, dt.minute
  // char logFileNameBuffer [50];
  // int cx ;


  boost::posix_time::ptime t = boost::posix_time::microsec_clock::universal_time();
  // cx = snprintf(logFileNameBuffer,50,"Webcam_position_log-%d.csv",boost::posix_time::to_iso_extended_string(t));
  // logFile.open("Webcam_position_log.csv");

  logFile.open("Webcam_position_log-"+boost::posix_time::to_iso_extended_string(t)+".csv");
  logFile<<"Timestamp,TagID,x(m),y(m),Theta(rad),\n";
};

RelativeChilitags::~RelativeChilitags(void){
  logFile.close();
};


void RelativeChilitags::updateCornerMap(chilitags::TagCornerMap &newTags){
  if (newTags.size() >= 3 && tags.size() != 1) {
    // If tags size = 1 then we have found no chilitgas - for our operation we want at least 4:
    // 3 calibration tags (0,1,2) - these are checked to exist below
    // and our fourth tag will be the one we are locating in reference to it.
    std::map<int, chilitags::Quad>::iterator it;
    bool referenceTagsFound = true;
    // Check if tags 0,1,2 are in the image:
    for (int i = 0; i < 3; i++) {
      it = newTags.find(i);
      if (it == newTags.end()) {
        // std::cout<<"Did not find tag "<<i<<"\n";
        referenceTagsFound = false;
      };
    };
    if (referenceTagsFound) {
      // If our reference/calibration tags exist, then update internal map and calibration values
      tags = newTags;
      calcCalibrationFactors();
      mapUpdated = true;
      // std::cout<<"Updating Found Tags \n";
    }

  } else {
    mapUpdated = false;
  }

};

void RelativeChilitags::calcCalibrationFactors(void){
  // refactor this to find correct way of accessing first two variables.
  // Tags are arranged as:
  // 0 ---- 1
  // |
  // |
  // 2

  cv::Point2f diff1 = centreOfTag(tags[1]) - centreOfTag(tags[0]);
  cv::Point2f diff2 = centreOfTag(tags[2]) - centreOfTag(tags[0]);

  float r1 = atan2(diff1.y,diff1.x);
  // The 0-2 line is viewed as positive y difference, so we subtract pi/2 to look for the orientation we want
  float r2 = atan2(diff2.y,diff2.x) - M_PI/2;
  rotation = (r1+ r2)/2;

  cv::Point2f tag1 = offsetAndRotate(centreOfTag(tags[1]));
  cv::Point2f tag2 = offsetAndRotate(centreOfTag(tags[2]));

  // Could use a norm here to include offsets due to incorrect rotation
  // float x_pixel_dist = cv::norm(tag1);
  // float y_pixel_dist = cv::norm(tag2);
  xConv = CALIBRATION_X/abs(tag1.x);
  yConv = CALIBRATION_Y/abs(tag2.y);
};

cv::Point2f RelativeChilitags::centreOfTag(chilitags::Quad tagRep){
  cv::Mat_<cv::Point2f> tagCorners(tagRep);
  cv::Point2f result = cv::Point2f(0.0, 0.0);
  for (int i = 0; i < 4; ++i){
      result += tagCorners(i) * 0.25;
  }
  return result;
}


float RelativeChilitags::tagRelativeOrientation(chilitags::Quad tag){
  cv::Mat_<cv::Point2f> tagCorners(tag);
  //  0 - 1
  //  |   | ------> (this is the direction we require)
  //  3 - 2
  cv::Point2f top = offsetAndRotate(tagCorners(1)) - offsetAndRotate(tagCorners(0));
  float top_angle = atan2(top.y,top.x);

  cv::Point2f bottom = offsetAndRotate(tagCorners(2)) - offsetAndRotate(tagCorners(3));
  float bottom_angle = atan2(bottom.y,bottom.x);


  cv::Point2f left = offsetAndRotate(tagCorners(3)) - offsetAndRotate(tagCorners(0));
  float left_angle = atan2(left.y,left.x) + M_PI/2;


  cv::Point2f right = offsetAndRotate(tagCorners(2)) - offsetAndRotate(tagCorners(1));
  float right_angle = atan2(right.y,right.x) + M_PI/2;

  return (top_angle + bottom_angle + left_angle + right_angle)/4;
}

cv::Point2f RelativeChilitags::offsetAndRotate(cv::Point2f position){
  // Calculating pixel offset from the zero tag
  cv::Point2f tag_minus_offset = position - centreOfTag(tags[0]);
  cv::Mat tag_matrix =(  cv::Mat_<float>(2,1)<< tag_minus_offset.x, tag_minus_offset.y);

  // Rotate from u,v to x,y
  cv::Mat R_x = ( cv::Mat_<float>(2,2)<<
                  cos(rotation), -sin(rotation),
                  -sin(rotation), -cos(rotation)
                );
  cv::Mat res = R_x * tag_matrix;
  cv::Point2f result(res);
  return result;
}

cv::Point2f RelativeChilitags::scaleOffsetAndRotate(cv::Point2f position){
  // Offset and rotate a tag from the u,v axes to x,y and scale by measured distances
  cv::Point2f relative = offsetAndRotate(position);
  relative.x *= xConv;
  relative.y *= yConv;
  return relative;
}

cv::Point2f RelativeChilitags::tagRelativePosition(chilitags::Quad tag){
  // Offset and rotate a tag from the u,v axes to x,y and scale by measured distances
  return scaleOffsetAndRotate(centreOfTag(tag));
}

void RelativeChilitags::logNovelTagLocations(void){
  if (tags.size() > 3 && mapUpdated){
    // If less than 3 tags, cannot log as no novel tags found
    std::map<int, chilitags::Quad>::iterator it=tags.begin();
    cv::Point2f position;
    float orientation;
    // Skip tags 0,1,2
    std::advance(it,3);
    // Since iterator has advanced, now only looping through other tags:
    for (; it!=tags.end(); ++it) {
      // time_t rawtime;
      // struct tm *ptm;
      // time (&rawtime);
      // ptm = gmtime( &rawtime );
      // std::cout<<asctime(ptm)<<" Tag "<<it->first<<" Position: "<<tagRelativePosition(*it)<<"\n";

      // using namespace boost::posix_time;
      boost::posix_time::ptime t = boost::posix_time::microsec_clock::universal_time();
      // position = centreOfTag(it->second);
      position = tagRelativePosition(it->second);
      orientation = tagRelativeOrientation(it->second);
      logFile << boost::posix_time::to_iso_extended_string(t)<<"+00:00,"<<it->first<<","<<position.x<<","<<position.y<<","<<orientation<<",\n";
      // std::cout << boost::posix_time::to_iso_extended_string(t)<<"Z,"<<it->first<<","<<tagRelativePosition(*it).x<<","<<tagRelativePosition(*it).y<<",\n";
      // boost::posix_time::ptime t = boost::posix_time::microsec_clock::universal_time();
      // std::cout << boost::posix_time::to_iso_extended_string(t) << "Z\n";
      // std::cout<<ptm->tm_hour<<":"<<ptm->tm_min<<"Tag "<<it->first<<" Position: "<<tagRelativePosition(*it)<<"    ";
    }
    // std::cout<<"\n";
  }
}

int main(int argc, char* argv[])
{
    // Initialising input video - basic options
    int xRes = 640;
    int yRes = 480;
    int cameraIndex = 0;
    // Below options work for USB webcam
    // int xRes = 1280;
    // int yRes = 720;
    // int cameraIndex = 4;

    if (argc > 2) {
        xRes = std::atoi(argv[1]);
        yRes = std::atoi(argv[2]);
    }
    if (argc > 3) {
        cameraIndex = std::atoi(argv[3]);
    }

    // The source of input images
    cv::VideoCapture capture(cameraIndex);
    if (!capture.isOpened())
    {
        std::cerr << "Unable to initialise video capture." << std::endl;
        return 1;
    }
    capture.set(cv::CAP_PROP_FRAME_WIDTH, xRes);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, yRes);

    cv::Mat inputImage;

    // We need separate Chilitags if we want to compare find() with different
    // detection/tracking parameters on the same image

    RelativeChilitags r;
    // This one is the reference Chilitags
    chilitags::Chilitags detectedChilitags;
    detectedChilitags.setFilter(0, 0.0f);


    cv::namedWindow("DisplayChilitags");

    // Do we want to run and show the reference detection ?
    bool showReference = true;
    // Do we want to run and show the tracking-based detection ?


    char keyPressed;
    while ('q' != (keyPressed = (char) cv::waitKey(1))) {

        // toggle the processing, according to user input
        if (keyPressed == 'd') showReference = !showReference;
        capture.read(inputImage);
        cv::Mat outputImage = inputImage.clone();
        int64 startTime = cv::getTickCount();
        auto tags = detectedChilitags.find(inputImage);
        int64 endTime = cv::getTickCount();
        r.updateCornerMap(tags);
        r.logNovelTagLocations();
        // drawTags(outputImage, tags, startTime, endTime, true);
        logTagPosition(outputImage, tags, startTime, endTime);

        cv::imshow("DisplayChilitags", outputImage);
    }

    cv::destroyWindow("DisplayChilitags");
    capture.release();

    return 0;
}





void logTagPosition(
  cv::Mat outputImage,
  const chilitags::TagCornerMap &tags,
  int64 startTime,
  int64 endTime){
      // Print in green
    cv::Scalar COLOR = cv::Scalar(0, 255, 0);
    // loop through tags:
    for (const auto & tag : tags) {

        const cv::Mat_<cv::Point2f> corners(tag.second);
        // std::cout<<"Tag: "<< tag.first <<"\n";
        for (size_t i = 0; i < 4; ++i) {
            static const int SHIFT = 16;
            static const float PRECISION = 1<<SHIFT;
            // std::cout<< corners(i)<<"\n";
            cv::line(
                outputImage,
                PRECISION*corners(i),
                PRECISION*corners((i+1)%4),
                COLOR, 1, cv::LINE_AA, SHIFT);
            // removing the shift & precision stuff doesn't plot a correct square.
        }

        cv::Point2f center = 0.5f*(corners(0) + corners(2));
        if (tag.first == 42 ) {
          cv::putText(outputImage, cv::format("aaa%d", tag.first), center,
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, COLOR);
          /* code */
        }else{
          cv::putText(outputImage, cv::format("%d", tag.first), center,
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, COLOR);
        };

    }

    float processingTime = 1000.0f*((float) endTime - startTime)/cv::getTickFrequency();
    cv::putText(outputImage,
                cv::format("%dx%d %4.0f ms (press '%c' to toggle %s)",
                           outputImage.cols, outputImage.rows,
                           processingTime,
                           'd',
                           "simple detection"
                           ),
                cv::Point(32,32),
                cv::FONT_HERSHEY_SIMPLEX, 0.5f, COLOR);
}


void drawTags(
    cv::Mat outputImage,
    const chilitags::TagCornerMap &tags,
    int64 startTime,
    int64 endTime,
    bool detection
    ){
    cv::Scalar COLOR = detection ?
                       cv::Scalar(0, 0, 255) :
                       cv::Scalar(255, 0, 0);

    for (const auto & tag : tags) {

        const cv::Mat_<cv::Point2f> corners(tag.second);

        for (size_t i = 0; i < 4; ++i) {
            static const int SHIFT = 16;
            static const float PRECISION = 1<<SHIFT;
            cv::line(
                outputImage,
                PRECISION*corners(i),
                PRECISION*corners((i+1)%4),
                COLOR, detection ? 3 : 1, cv::LINE_AA, SHIFT);

        }

        cv::Point2f center = 0.5f*(corners(0) + corners(2));
        if (tag.first == 42 ) {
          cv::putText(outputImage, cv::format("aaa%d", tag.first), center,
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, COLOR);
          /* code */
        }else{
          cv::putText(outputImage, cv::format("%d", tag.first), center,
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, COLOR);
        };

    }

    float processingTime = 1000.0f*((float) endTime - startTime)/cv::getTickFrequency();
    cv::putText(outputImage,
                cv::format("%dx%d %4.0f ms (press '%c' to toggle %s)",
                           outputImage.cols, outputImage.rows,
                           processingTime,
                           detection ? 'd' : 't',
                           detection ? "simple detection" : "tracking"
                           ),
                cv::Point(32,detection ? 32 : 64),
                cv::FONT_HERSHEY_SIMPLEX, 0.5f, COLOR);
}
