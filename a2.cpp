//// B657 assignment 2 skeleton code
//
// Compile with: "make"
//
// See assignment handout for command line and project specifications.


//Link to the header file
#include "CImg.h"
#include <ctime>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <Sift.h>
#include<math.h>
#include<dirent.h>


//Use the cimg namespace to access the functions easily
using namespace cimg_library;
using namespace std;

//function to calculate the best features based on distance matching
float comparingvalues(CImg<double> input_image3,CImg<double> input_image1,vector<SiftDescriptor> &descriptorsip,vector<SiftDescriptor> &descriptors,int taskno)
{
           int count=0;
           const unsigned char color[3] = {0,0,255};
           float best,secondbest;int bestindex,secondindex;
           vector<float> distance;
           for(int i=0;i<descriptorsip.size();i++)
          {
             for(int j=0;j<descriptors.size();j++)
             {
                int sum=0;
                //sum of all the differences in euclidian distance of descriptors
                for(int l=0; l<128; l++)
                {
                    float diff=(descriptorsip[i].descriptor[l])-(descriptors[j].descriptor[l]);
                    sum= sum +(diff*diff);
                   
                }
               float distance=sqrt(sum);
               //keep initial distance as best
               if(j==0){
                   best=distance;secondbest=distance;bestindex=j;secondindex=j;}
               else{
                if (distance<secondbest && distance>best)
               {
                    secondbest=distance;
                    secondindex=j;
                 
               }
               else if(distance<best)
               {
                   best=distance;
                   bestindex=j;
               }
               else
                   continue;
               }
            }
             //matching with threshold value
           if((best/secondbest)<0.8 ){
               if(taskno==1){
                   input_image3.draw_line(descriptorsip[i].col,descriptorsip[i].row,input_image1.width()+(descriptors[bestindex].row/2),descriptors[bestindex].col,color,1);
                   //drawing descriptors
                   for(int j=0; j<5; j++)
	            for(int k=0; k<5; k++)
		      if(j==2 || k==2)
		        for(int p=0; p<3; p++)
                          if(descriptorsip[i].col+k < input_image1.width() && descriptorsip[i].row+j < input_image1.height())
                                  input_image1(descriptorsip[i].col+k, descriptorsip[i].row+j, 0, p)=0;
               }
               else
                   count =count+1;
           }
           }
           if(taskno==1)
               input_image3.get_normalize(0,255).save("sift.png"); 
           return count;
}


void drawSingleImage(vector<SiftDescriptor> &descriptors1,vector<SiftDescriptor> &descriptors2,CImg<double> input_image3,CImg<double> input_image1,CImg<double> input_image2 )
{
    int val=comparingvalues(input_image3,input_image1,descriptors1,descriptors2,1);
}

//sorting images  based on feature matches and displaying list
void sortImages(vector<string> fileNames,CImg<double> queryimg,vector<SiftDescriptor> &descriptorsip)
{
     cout<<endl<<"Second step";
     std::vector<float> counts;
     std::vector<int> indexes;
     //loop over images to find features
     for(int k=0;k<fileNames.size();k++)
     {
         CImg<double> input_image(fileNames[k].c_str());
         //convert to greyscale
         CImg<double> gray = input_image.get_RGBtoHSI().get_channel(2);
         vector<SiftDescriptor> descriptors = Sift::compute_sift(gray);
         //get the count of matching features
         int count =comparingvalues(queryimg,queryimg,descriptorsip,descriptors,2);
         
       //  store the counts to keep track
         counts.push_back(count);
         indexes.push_back(count);
     }
     sort(counts.begin(),counts.end());
    // display the sorted list of images based on  matched feature 
     for (int i=counts.size()-1;i>=0;i--) {
         int pos=find(indexes.begin(),indexes.end(),counts[i])-indexes.begin();
         if(pos>=indexes.size())
             cout<<"bad index matching for same attraction";
         else
             cout<<endl<<fileNames[pos]<<endl;
      }
}


// finding the precision value for all images that are correctly matched from the same attraction
void retrievalAlgorithm(vector<string> fileNames,string queryimg,vector<SiftDescriptor> &descriptorsip)
{
    cout<<endl<<"Third step"<<endl;
//    DIR *folder;
//    dirent *pdr;
//    string path="part1_images\\";
//    vector<string> filenames;
//    folder=opendir(path.c_str());
//    if(!folder)
//        cout<<"no dir";
//    else
//    {
//    while(pdr=readdir(folder))
//    {
//        filenames.push_back(pdr->d_name);
//    }
//    for(int i=0;i<filenames.size();i++)
//    {
//        cout<<filenames[i]<<endl;
//    }}
     std::vector<float> counts;
     std::vector<int> indexes;
     //loop over images to find features
     for(int k=0;k<fileNames.size();k++)
     {
         CImg<double> input_image(fileNames[k].c_str());
         CImg<double> query_image(queryimg.c_str());
         //convert to greyscale
         CImg<double> gray = input_image.get_RGBtoHSI().get_channel(2);
         vector<SiftDescriptor> descriptors = Sift::compute_sift(gray);
         
         int count=comparingvalues(query_image,query_image,descriptorsip,descriptors,2);
       //  store the counts to keep track
         counts.push_back(count);
         indexes.push_back(count);
     }
     sort(counts.begin(),counts.end());
     float precision=0;
     std::string::size_type pos=queryimg.find('_');
     std::string queryim;
     if(pos!=std::string::npos)
         queryim=queryimg.substr(0,pos);
    // display the sorted list of images based on  matched feature 
     for (int i=counts.size()-1;i>counts.size()-11;i--) {
         int pos=find(indexes.begin(),indexes.end(),counts[i])-indexes.begin();
         if(pos>=indexes.size())
             cout<<"error";
         else
         {
             cout<<fileNames[pos]<<endl;
             if(fileNames[pos].find(queryim)!=std::string::npos)
                 precision=precision+1;
             else
                 cout<<"not found"<<endl;
         }
      }
     cout<<"precisions is : "<<(precision/10.0);
}

//generating gaussian vector
void findGaussian()
{ 
   // std::normal_distribution<> d(5,2);
    std::vector<float> values;
    
    for(int n=0; n<128; n++) {
        values.push_back((double)rand() / (RAND_MAX + 1.0));
    }
    for(int i=values.size()-1;i>=0;i--)
        cout<<values[i]<<endl;
    
}

//warping function
// define the inverse warping function 
CImg<double> inverse_warp ( CImg<double> &input)
{ 
int w=input._width ;
int	l=input._height ;
CImg<double> output(input,0);

int i,j,k,m ;
double h[3][3];
h[1][1] = 1.12 ;
h[1][2] = -0.31 ;
h[1][3] = 223 ;
h[2][1] = 0.11 ;
h[2][2] = 0.69 ;
h[2][3] = -19.92 ;
h[3][1] = 0.00026 ;
h[3][2] = -0.000597 ;
h[3][3] = 1 ;

	for(int i=0 ; i<l ; i++)
		{ for(int j=0 ; j<w ; j++ )
		{  
			k= round(h[1][1]*j+h[1][2]*i+h[1][3] );
		 	m= round (h[2][1]*j+h[2][2]*i+h[2][3] );
		 	if ( (0<k && k<=l) || (0<m && m<=w) )
		  		{
				output(i,j)=input(k,m) ; }
			else 
				{ output(i,j) = 255 ; 
				}
		  
		}
		}
return output ; 

}

int main(int argc, char **argv)
{
  try {

    if(argc < 2)
      {
	cout << "Insufficent number of arguments; correct usage:" << endl;
	cout << "    a2-p1 part_id ..." << endl;
	return -1;
      }
    //storing the command parameters which are filenames
    std::vector<std::string> fileNames;
    string part = argv[1];
    string inputFile1 = argv[2];
    for(int i=0;i<=argc-4;i++)
    {
        fileNames.push_back(argv[3+i]);
    }
    string inputFile2=argv[3];
    if(part == "part1")
      {
	// This is just a bit of sample code to get you started, to
	// show how to use the SIFT library.

	CImg<double> input_image1(inputFile1.c_str());
        CImg<double> input_image2(inputFile2.c_str());
        
        
	// convert image to grayscale
	CImg<double> gray1 = input_image1.get_RGBtoHSI().get_channel(2);
        CImg<double> gray2= input_image2.get_RGBtoHSI().get_channel(2);
        vector<SiftDescriptor> descriptors1 = Sift::compute_sift(gray1);
        vector<SiftDescriptor> descriptors2 = Sift::compute_sift(gray2);
               
        CImg<double> input_image3(input_image1.append(input_image2));
      // if arguments are exactly 4
       if (argc==4)
          drawSingleImage(descriptors1,descriptors2,input_image3,input_image1,input_image2);
  
       if(argc>4){
            sortImages(fileNames,input_image1,descriptors1);
            retrievalAlgorithm(fileNames,inputFile1,descriptors1);
            
            findGaussian();
       }
	for(int i=0; i<descriptors1.size(); i++)
	  {
//	    cout << "Descriptor #" << i << ": x=" << descriptors1[i].col << " y=" << descriptors1[i].row << " descriptor=(";
//	    for(int l=0; l<128; l++)
//	      cout << descriptors1[i].descriptor[l] << "," ;
//	    cout << ")" << endl;
//
//	    for(int j=0; j<5; j++)
//	      for(int k=0; k<5; k++)
//		if(j==2 || k==2)
//		    for(int p=0; p<3; p++)
//                    if(descriptors[i].col+k < input_image.width() && descriptors[i].row+j < input_image.height())
//                      input_image(descriptors[i].col+k, descriptors[i].row+j, 0, p)=0;
//
//
//	  }      
	
        }}
    else if(part == "part2")
      {
	 //call the inverse warping function
         CImg<double> input_image(inputFile1.c_str());
         CImg<double> lincoln_warped= inverse_warp(input_image) ;
      			   lincoln_warped.save("lincoln_warped.png") ; 	
        
      }
    else
      throw std::string("unknown part!");

    // feel free to add more conditions for other parts (e.g. more specific)
    //  parts, for debugging, etc.
  }
  catch(const string &err) {
    cerr << "Error: " << err << endl;
  }
}









