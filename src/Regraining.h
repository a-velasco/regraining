#include <vector>
#include "opencv2/imgproc.hpp"

template< class TImage >
class Regraining
{
  public:
    static TImage Regrain( TImage originalImage,
                           TImage colorMatchedImage,
                           std::vector<int> & nbits,
                           int level,
                           int minDimension = 20)
    {
      // Initial images must be 3 channel double, from 0-1
      originalImage.convertTo( originalImage, CV_64FC3, 1.f/255 );
      colorMatchedImage.convertTo( colorMatchedImage, CV_64FC3, 1.f/255 );
      
      TImage regrainedImage = originalImage; // Initialize regrainedImage to originalImage
      Downsample(regrainedImage, originalImage, colorMatchedImage, nbits, level);

      return regrainedImage;
    }

  private:
    static std::vector< TImage > ShiftImage( const TImage & img )
    {
      typedef TImage ImageType;

      cv::Range leftCols(0,img.cols-1), rightCols(1,img.cols);
      cv::Range upperRows(0,img.rows-1), lowerRows(1,img.rows);

      int lastRow = img.rows-1;
      int lastCol = img.cols-1;

      ImageType shiftLeft; // shift by one column to the left
      ImageType mainBodyLeft = img( cv::Range::all(), leftCols );
      ImageType colToAppendLeft( img.rows, 1, img.type() );
      img.col(0).copyTo( colToAppendLeft.col(0) );
      cv::hconcat( colToAppendLeft, mainBodyLeft, shiftLeft );

      ImageType shiftRight; // shift by one column to the right
      ImageType mainBodyRight = img( cv::Range::all(), rightCols );
      ImageType colToAppendRight( img.rows, 1, img.type() );
      img.col(lastCol).copyTo( colToAppendRight.col(0) );
      cv::hconcat( mainBodyRight, colToAppendRight, shiftRight );

      ImageType shiftUp; // shift by one row upward
      ImageType mainBodyUp = img( upperRows, cv::Range::all() );
      ImageType rowToAppendUp( 1, img.cols, img.type() );
      img.row(0).copyTo( rowToAppendUp.row(0) );
      cv::vconcat( rowToAppendUp, mainBodyUp, shiftUp );

      ImageType shiftDown; // shift by one row downward
      ImageType mainBodyDown = img( lowerRows, cv::Range::all() );
      ImageType rowToAppendDown( 1, img.cols, img.type() );
      img.row(lastRow).copyTo( rowToAppendDown.row(0) );
      cv::vconcat( mainBodyDown, rowToAppendDown, shiftDown );

      // Order: Left, Right, Up, Down
      std::vector< ImageType > shiftedImages = { shiftLeft, shiftRight, shiftUp, shiftDown };

      return shiftedImages;
    }

    static void Solve(TImage & regrainedImage,
               const TImage & originalImage,
               const TImage & colorMatchedImage,
               const int & nbits,
               const int & level )
    {
      typedef TImage ImageType;

      // G0 = I
      std::vector< ImageType > originalImageShifted = ShiftImage( originalImage );

      // Gradient matrices
      ImageType G0x = originalImageShifted[1] - originalImageShifted[0];
      ImageType G0y = originalImageShifted[3] - originalImageShifted[2];

      // Sum each gradient image along its 3 channels
      std::vector< ImageType > G0xChannels(3);
      cv::split( G0x, G0xChannels );
      ImageType G0xC1 = G0xChannels[0] + G0xChannels[1] + G0xChannels[2];
      G0xC1.convertTo( G0xC1, CV_64FC1 );

      std::vector< ImageType > G0yChannels(3);
      cv::split( G0y, G0yChannels );
      ImageType G0yC1 = G0yChannels[0] + G0yChannels[1] + G0yChannels[2];
      G0yC1.convertTo( G0yC1, CV_64FC1 );

      // Compute gradient image dI
      ImageType G0x2 = G0xC1.mul( G0xC1 );
      ImageType G0y2 = G0yC1.mul( G0yC1 );

      ImageType gradOriginalImage( G0xC1.rows, G0xC1.cols, G0xC1.type() );
      cv::sqrt( (G0x2+G0y2), gradOriginalImage );

      ImageType psi = 255 * gradOriginalImage.clone();
      for( int i = 0; i < psi.rows; i++ )
      {
        for( int j = 0; j < psi.cols; j++ )
        {
          if( psi.template at<double>(i,j) > 5. )
          {
            psi.template at<double>(i,j) = 1.;
          }
          else
          {
            psi.template at<double>(i,j) = psi.template at<double>(i,j) / 5.;
          }
        }
      }

      psi.convertTo( psi, CV_64FC1 );

      std::vector< ImageType > psi_channels = { psi, psi, psi };
      
      ImageType psiC3;
      cv::merge( psi_channels, psiC3 );

      // Phi = one for now, can't get it to work well with paper's proposed function
      ImageType phi = ImageType( gradOriginalImage.rows, gradOriginalImage.cols, CV_64FC1, cv::Scalar(1.,1.,1.) );

      ImageType originalImageLeft  = originalImageShifted[0].clone();
      ImageType originalImageRight = originalImageShifted[1].clone();
      ImageType originalImageUp    = originalImageShifted[2].clone();
      ImageType originalImageDown  = originalImageShifted[3].clone();

      double rho = 1./5.;
      ImageType term1 = psiC3.mul(colorMatchedImage);
      ImageType term2, term3, term4, term5;
      for( int i = 0; i < nbits; i++ )
      {
        ImageType den = psi + ( 4 * phi );

        // Calculate current displaced regrainedImage matrices
        ImageType regrainedImageLeft, regrainedImageRight, regrainedImageUp, regrainedImageDown;
        std::vector< ImageType > regrainedImageShifted = ShiftImage( regrainedImage );
        regrainedImageLeft  = regrainedImageShifted[0];
        regrainedImageRight = regrainedImageShifted[1];
        regrainedImageUp    = regrainedImageShifted[2];
        regrainedImageDown  = regrainedImageShifted[3];

        term2 = regrainedImageRight - originalImageRight + originalImage;
        term3 = regrainedImageDown - originalImageDown + originalImage;
        term4 = regrainedImageLeft - originalImageLeft + originalImage;
        term5 = regrainedImageUp - originalImageUp + originalImage;

        ImageType num = term1 + term2 + term3 + term4 + term5;

        std::vector< ImageType > denVec = { den, den, den };

        ImageType denC3;
        cv::merge( denVec, denC3 );

        ImageType regrainedImage_temp;
        cv::divide( num, denC3, regrainedImage_temp );
        regrainedImage = regrainedImage_temp * (1 - rho) + (rho * regrainedImage);
      }
    }

    static void Downsample( TImage &regrainedImage,
                     const TImage &originalImage,
                     const TImage &colorMatchedImage,
                     std::vector<int> & nbits,
                     int level,
                     int minDimension = 20 )
    {
      typedef TImage ImageType;

      std::vector<int>::const_iterator first = nbits.begin();
      std::vector<int>::const_iterator last = nbits.end();

      ImageType regrainedImage2( regrainedImage.rows, regrainedImage.cols, CV_64FC3 );
      ImageType originalImage2( originalImage.rows, originalImage.cols, CV_64FC3 );
      ImageType colorMatchedImage2( colorMatchedImage.rows, colorMatchedImage.cols, colorMatchedImage.type() );

      if( ( nbits.size() > 1 ) && ( std::ceil(originalImage.cols/2) > minDimension) && ( std::ceil(originalImage.rows/2) > minDimension ) )
      {
        cv::Size downsampleSize = cv::Size(std::ceil(originalImage.cols/2), std::ceil(originalImage.rows/2));
        cv::resize( originalImage,  originalImage2, downsampleSize );
        cv::resize( colorMatchedImage, colorMatchedImage2, downsampleSize );
        cv::resize( regrainedImage,  regrainedImage2,  downsampleSize );

        std::vector<int> new_nbits(first+1, last);

        Downsample( regrainedImage2, originalImage2, colorMatchedImage2, new_nbits, level+1 );
        cv::resize( regrainedImage2, regrainedImage, cv::Size(originalImage.cols, originalImage.rows) );
      }

      Solve( regrainedImage, originalImage, colorMatchedImage, nbits[0], level );
    }

};