#include <math.h>
#include <iostream>

#include "Regraining.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

///////////////////////////////////////////////////////////////////////////////////////////////
// Implementation of regraining algorithm as described in:                                   //
// [Pitie07a] Automated colour grading using colour distribution transfer.                   //
//                                                                                           //
// F. Pitie , A. Kokaram and R. Dahyot (2007) Computer Vision and Image                      //
// Understanding.                                                                            //
//                                                                                           //
// Inspired by the author's MATLAB implementation at https://github.com/frcs/colour-transfer //
///////////////////////////////////////////////////////////////////////////////////////////////

int main( int argc, char* argv[] )
{
  // Check inputs
  if( argc != 4 )
  {
    std::cout << "Usage:\n" << "argv[1]: (IN)  Original (before histogram matching) image path\n" 
                            << "argv[2]: (IN)  Color matched image path\n"
                            << "argv[3]: (OUT) Regrained image path\n";
    return 1;
  }

  typedef cv::Mat ImageType;

  std::string originalImageFileName = argv[1];
  std::string colorMatchedImageFileName = argv[2];
  std::string outputImageFileName = argv[3];

  // Read original image
  ImageType originalImage = cv::imread(originalImageFileName);
  if(! originalImage.data )
  {
    std::cout <<  "Could not open or find the image" << std::endl ;
    return 1;
  }

  // Read color matched image
  ImageType colorMatchedImage = cv::imread(colorMatchedImageFileName);
  if(! colorMatchedImage.data )
  {
    std::cout <<  "Could not open or find the image" << std::endl ;
    return 1;
  }

  std::vector< int > nbits = {4, 16, 32, 64, 64, 64};

  // Solve
  ImageType regrainedImage = Regraining< ImageType >::Regrain( originalImage, colorMatchedImage, nbits, 0 );

  // Create windows for display
  cv::namedWindow( "Original", cv::WINDOW_AUTOSIZE );
  cv::namedWindow( "Colormatched", cv::WINDOW_AUTOSIZE );
  cv::namedWindow( "Regrained", cv::WINDOW_AUTOSIZE );

  cv::imshow( "Original", originalImage);
  cv::imshow( "Colormatched", colorMatchedImage);
  cv::imshow( "Regrained", regrainedImage);
  cv::waitKey(0);

  // Save result
  regrainedImage.convertTo( regrainedImage, CV_64FC3, 255. );

  cv::imwrite( outputImageFileName, regrainedImage );

}