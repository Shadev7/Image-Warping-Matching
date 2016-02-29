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
#include <sstream>
#include <limits> 
#include <utility>

//Use the cimg namespace to access the functions easily
using namespace cimg_library;
using namespace std;

//function to calculate the best features based on distance matching

float comparingvalues(CImg<double> input_image3, CImg<double> input_image1, vector<SiftDescriptor> &descriptorsip, vector<SiftDescriptor> &descriptors, int taskno,int bestin) {
    int count = 0;
    const unsigned char color[3] = {0, 0, 255};
    float best, secondbest;
    int bestindex, secondindex;
    vector<float> distance;
    for (int i = 0; i < descriptorsip.size(); i++) {
        for (int j = 0; j < descriptors.size(); j++) {
            int sum = 0;
            //sum of all the differences in euclidian distance of descriptors
            for (int l = 0; l < 128; l++) {
                float diff = (descriptorsip[i].descriptor[l])-(descriptors[j].descriptor[l]);
                sum = sum + (diff * diff);

            }
            float distance = sqrt(sum);
            //keep initial distance as best
            if (j == 0) {
                best = distance;
                secondbest = distance;
                bestindex = j;                
                secondindex = j;
            } else {
                if (distance < secondbest && distance > best) {
                    secondbest = distance;
                    secondindex = j;

                } else if (distance < best) {
                    best = distance;
                    bestindex = j;
                } else
                    continue;
            }
        }
        //matching with threshold value
        if ((best / secondbest) < 0.8) {
            if (taskno == 1) {
                input_image3.draw_line(descriptorsip[i].col, descriptorsip[i].row, input_image1.width()+(descriptors[bestindex].col), descriptors[bestindex].row, color, 1);
                //drawing descriptors
                for (int j = 0; j < 5; j++)
                    for (int k = 0; k < 5; k++)
                        if (j == 2 || k == 2)
                            for (int p = 0; p < 3; p++)
                                if (descriptorsip[i].col + k < input_image1.width() && descriptorsip[i].row + j < input_image1.height())
                                    input_image3(descriptorsip[i].col + k, descriptorsip[i].row + j, 0, p) = 0;
            } else
                count = count + 1;
        }        
    }
    if (taskno == 1)
        input_image3.get_normalize(0, 255).save("sift.png");
    return count;
}

void drawSingleImage(vector<SiftDescriptor> &descriptors1, vector<SiftDescriptor> &descriptors2, CImg<double> input_image3, CImg<double> input_image1, CImg<double> input_image2) {
    int val = comparingvalues(input_image3, input_image1, descriptors1, descriptors2, 1,-1);
}

//sorting images  based on feature matches and displaying list

void sortImages(vector<string> fileNames, CImg<double> queryimg, vector<SiftDescriptor> &descriptorsip) {
    cout << endl << "Second step";
    std::vector<float> counts;
    std::vector<int> indexes;
    //loop over images to find features
    for (int k = 0; k < fileNames.size(); k++) {
        CImg<double> input_image(fileNames[k].c_str());
        //convert to greyscale
        CImg<double> gray = input_image.get_RGBtoHSI().get_channel(2);
        vector<SiftDescriptor> descriptors = Sift::compute_sift(gray);
        //get the count of matching features
        int count = comparingvalues(queryimg, queryimg, descriptorsip, descriptors, 2,-1);

        //  store the counts to keep track
        counts.push_back(count);
        indexes.push_back(count);
    }
    sort(counts.begin(), counts.end());
    // display the sorted list of images based on  matched feature 
    for (int i = counts.size() - 1; i >= 0; i--) {
        int pos = find(indexes.begin(), indexes.end(), counts[i]) - indexes.begin();
        if (pos >= indexes.size())
            cout << "bad index matching for same attraction";
        else
            cout << endl << fileNames[pos] << endl;
    }
}


// finding the precision value for all images that are correctly matched from the same attraction

void retrievalAlgorithm(vector<string> fileNames, string queryimg, vector<SiftDescriptor> &descriptorsip) {
    cout << endl << "Third step" << endl;
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
    for (int k = 0; k < fileNames.size(); k++) {
        CImg<double> input_image(fileNames[k].c_str());
        CImg<double> query_image(queryimg.c_str());
        //convert to greyscale
        CImg<double> gray = input_image.get_RGBtoHSI().get_channel(2);
        vector<SiftDescriptor> descriptors = Sift::compute_sift(gray);

        int count = comparingvalues(query_image, query_image, descriptorsip, descriptors, 2,-1);
        //  store the counts to keep track
        counts.push_back(count);
        indexes.push_back(count);
    }
    sort(counts.begin(), counts.end());
    float precision = 0;
    std::string::size_type pos = queryimg.find('_');
    std::string queryim;
    if (pos != std::string::npos)
        queryim = queryimg.substr(0, pos);
    // display the sorted list of images based on  matched feature 
    for (int i = counts.size() - 1; i > counts.size() - 11; i--) {
        int pos = find(indexes.begin(), indexes.end(), counts[i]) - indexes.begin();
        if (pos >= indexes.size())
            cout << "error";
        else {
            cout << fileNames[pos] << endl;
            if (fileNames[pos].find(queryim) != std::string::npos)
                precision = precision + 1;
            else
                cout << "not found" << endl;
        }
    }
    cout << "precisions is : " << (precision / 10.0);
}

//generating gaussian vector

int findGaussian(string queryimgimg,string fileName, vector<SiftDescriptor> &descriptorsip,vector<SiftDescriptor> &descriptors,CImg<double> queryimg) {
    vector< vector<double> > dim_query_matrix(descriptorsip.size(),vector<double> (10));
    vector< vector<double> > dim_sample_matrix(descriptors.size(),vector<double> (10));
    vector<vector<double> > values(10,vector<double> (128));
    float best, secondbest,bestreal,secndbestreal;
    int bestindex, secondindex,bestindreal,secbestindexreal;
    int count=0,ct=0;
    //creating the gaussian 128D random matrix of dimension 10
    for(int k=0;k<10;k++)
    {
        std::vector<int> descriptor;
        //generating the gaussian random vectors
         for (int n = 0; n < 128; n++) {
           double a= ((double) rand() / (RAND_MAX + 1.0));
           values[k][n]=a;
        }       
    }
    //finding the dot product of gaussian vector and descriptors of query image along dimensions
    for (int i = 0; i < descriptorsip.size(); i++) {
            for(int j=0;j<10;j++)
            {
              int sum = 0;
              for(int k=0;k<128;k++)
              {
                  sum= sum + descriptorsip[i].descriptor[k] * values[j][k];
              }
              int dotpdt=sum/150;
              dim_query_matrix[i][j]=dotpdt;
            }
        }
    //finding the dot product of gaussian vector and descriptors of example image along dimensions
    for(int i=0;i<descriptors.size();i++)
    {
        for(int j=0;j<10;j++)
        {
             int sum = 0;
              for(int k=0;k<128;k++)
              {
                  sum= sum + descriptors[i].descriptor[k] * values[j][k];
              }
              int dotpdt=sum/150;
              dim_sample_matrix[i][j]=dotpdt;
            //  cout<<dim_sample_matrix[i][j]<<endl;
        }
    }
    cout<<"find gaussian";
    //finding the closest summary vectors
    vector< int>  bestimgindexes ;
    vector< int>  bestsmplindexes ;
    int l=0;
    for(int i=0;i<dim_query_matrix.size();i++)
    {
         int count=0;
        for(int j=0;j<dim_sample_matrix.size();j++)
        {
            int sum=0;
            for(int k=0;k<10;k++)
            {
                float diff = (dim_query_matrix[i][k]- dim_sample_matrix[j][k]);
                sum=sum + (diff*diff);
            }
           float distance = sqrt(sum);
         //  cout<<"distance : " <<distance<<endl;
         //keep initial distance as best
           if (j == 0) {
                best = distance;
                secondbest = distance;
                bestindex = j;                
                secondindex = j;
            } else {
                if (distance < secondbest && distance > best) {
                    secondbest = distance;
                    secondindex = j;

                } else if (distance < best) {
                    best = distance;
                    bestindex = j;
                } else
                    continue;
            }}         
          if(best/secondbest<0.8)
          {
            bestimgindexes.push_back(i);
            bestsmplindexes.push_back(bestindex);
           }
    }
    cout<<"count of l " <<bestimgindexes.size() << endl;
//    for(int i=0;i<bestindexes.size();i++)
//    {
//        cout<<bestindexes[i][0];
//    }

    for (int i = 0; i < bestimgindexes.size(); i++) {
        for(int j=0;j<bestimgindexes.size();j++){
               int sum=0;
               for (int m = 0; m < 128; m++) {
                 float diff = (descriptorsip[bestimgindexes[i]].descriptor[m])-(descriptors[bestsmplindexes[i]].descriptor[m]);
                 sum = sum +(diff*diff);  
               } 
               float dist = sqrt(sum);
               //cout<<dist<<endl;
            //keep initial distance as best
               if(dist==0)
                   ct =ct+1;
        }
    }
       return ct;
    }

void ExactApprox(vector<string> fileNames,string queryimg, vector<SiftDescriptor> &descriptorsip)
{
    std::vector<float> counts;
    std::vector<int> indexes;
    for (int k = 0; k < fileNames.size(); k++) {
        CImg<double> input_image(fileNames[k].c_str());
        //convert to greyscale
        CImg<double> gray = input_image.get_RGBtoHSI().get_channel(2);
        vector<SiftDescriptor> descriptors = Sift::compute_sift(gray);
        int count = findGaussian(queryimg, fileNames[k], descriptorsip, descriptors,input_image);
        cout<<"count for "<<k<<" "<<count<<endl;
        //  store the counts to keep track
        counts.push_back(count);
        indexes.push_back(count);
    }
    sort(counts.begin(), counts.end());
    float precision = 0;
    std::string::size_type pos = queryimg.find('_');
    std::string queryim;
    if (pos != std::string::npos)
        queryim = queryimg.substr(0, pos);
    // display the sorted list of images based on  matched feature 
    for (int i = counts.size() - 1; i > counts.size() - 11; i--) {
        int pos = find(indexes.begin(), indexes.end(), counts[i]) - indexes.begin();
        if (pos >= indexes.size())
            cout << "error";
        else {
            cout << fileNames[pos] << endl;
            if (fileNames[pos].find(queryim) != std::string::npos)
                precision = precision + 1;
            else
                cout << "not found" << endl;
        }
    }
    cout << "precisions is : " << (precision / 10.0);
    }

   
    
//Part 2 2
double featureDistance(SiftDescriptor idescriptor, SiftDescriptor cdescriptor)
{
	double dist= 0;
	
	for(int l=0; l<128; l++)
	{
		
		dist += pow((idescriptor.descriptor[l] - cdescriptor.descriptor[l]),2);
	}
	dist = sqrt(dist);
	return dist;
}

int getMatchingFeature (vector <double> dRow, int features)
{
	int closestnbr;
	double val_closestnbr = std::numeric_limits<double>::max();
	double val_scndclosestnbr = std::numeric_limits<double>::max();
	double matchratio;
	double threshold = 0.8;
	//int closestneighbour;
	/* If current element is smaller than first 
           then update both first and second */
	for (int i = 0; i <features ; i++)
	{
        if (dRow[i] < val_closestnbr)
        {
            val_scndclosestnbr = val_closestnbr;
            val_closestnbr = dRow[i];
			closestnbr = i;
        }
 
        /* If dRow[i] is in between first and second 
           then update second  */
        else if (dRow[i] < val_scndclosestnbr && dRow[i] != val_closestnbr)
            val_scndclosestnbr = dRow[i];
	}
	//cout << "closest : "<<val_closestnbr << endl;
	//cout << "second : "<<val_scndclosestnbr << endl;
	//cout << "neighbour : "<<closestnbr << endl;
	
	matchratio = val_closestnbr / val_scndclosestnbr;
	
	if (matchratio < threshold)
		return closestnbr;
	else
		return 0;
}

vector<pair<int,int> > allFeatureDistance(vector<SiftDescriptor> idescriptors, vector<SiftDescriptor> cdescriptors)
{
	int no_i_dec = idescriptors.size();
	int no_c_desc = cdescriptors.size(); 
	
	//cout << "IP : "<<no_i_dec << endl;
	//cout << "Other : "<<no_c_desc << endl;
	double dist;  // = featureDistance (idescriptors[0],cdescriptors[0]);
	
	//vector< vector<double> > allDistance;
	pair<int,int> match; // matching features
    vector<pair<int,int> > matches; // all matching features
	vector <double> dRow(1,no_c_desc) ;
	int matchelem;
	
	for (int i=0 ; i< no_i_dec; i ++)
	{
		//vector <double> dRow(1,no_c_desc) ;
		
		for (int j=0 ; j< no_c_desc; j ++)
		{
			dist = featureDistance (idescriptors[i],cdescriptors[j]);
			dRow.push_back(dist);
		}
		//double result = std::min_element(std::begin(dRow), std::end(dRow));
		//allDistance.push_back(dRow);
		
		
		// find matches
		matchelem = getMatchingFeature ( dRow, no_c_desc);
		if (matchelem != 0)
		{
			match.first = i;
			match.second = matchelem;
			matches.push_back(match);
		}
		dRow.clear();
	}
	cout << " tot Match : " << matches.size() << endl ;
	return matches;
	//matches.clear();
	//print matching elements
	/* for (int j=0 ; j< matches.size(); j ++)
	{
		cout << "Match : " << matches[j].first << " , " << matches[j].second << endl ;
	} */
	
	
	/*for (int j=0 ; j< no_c_desc; j ++)
		{
			cout << dRow[j] << "," ;
		}
		//double result = std::min_element(std::begin(dRow), std::end(dRow));
		//cout << "MIN : "<<result << endl;
		int matchelem = getMatchingFeature ( dRow, no_c_desc);
		if (matchelem != 0)
		{
			match.first = 0;
			match.second = matchelem;
			matches.push_back(match);
		}*/
	//cout << "Match : " << matches[0].first << " , " << matches[0].second ;
		//cout <<endl ;
	//cout << dist << endl;

}

CImg <double> findTransformationMatrix(vector<pair<int,int> > matches,vector<SiftDescriptor> descriptors,vector<SiftDescriptor> cmpdescriptors){
	int min = 0;
	int max = matches.size();
	int randNum ;
	int noOfSamples = 4;
	double x1, y1, x2, y2;
	//pair<int,int> current;
	vector<vector<double> > ip;
	vector<double> temp;
	vector<double> op;
	CImg <double> A (8,8,1,1);
	//CImg <double> A (8,8);
	CImg <double> B (1,8,1,1);
	CImg <double> Tr (1,8,1,1);
	int random_match_ip;
	int random_match_op;
	;
	//for (int i = 0; i< 1000; i++){
		for (int j = 0; j< noOfSamples ; j++)
		{
			randNum = rand()%(max-min + 1) + min;
			random_match_ip = matches[randNum].first;
			random_match_op = matches[randNum].second;
			x1 = descriptors[random_match_ip].col;
			y1 = descriptors[random_match_ip].row;
			
			x2 = cmpdescriptors[random_match_op].col;
			y2 = cmpdescriptors[random_match_op].row;
			//cout << "x1 " << x1 << " y1 " << y1 << " x2 " << x2 << " y2 " << y2 << endl;
						
			
			/*A (0+j,0+j,1,1,0) = x1;
			A (0+j,1+j,1,1,0) = y1;
			A (0+j,2+j,1,1,0) = 1;
			A (0+j,3+j,1,1,0) = 0;
			A (0+j,4+j,1,1,0) = 0;
			A (0+j,5+j,1,1,0) = 0;
			A (0+j,6+j,1,1,0) = -(x1*x2);
			A (0+j,7+j,1,1,0) = -(y1*x2);
			
			A (1+j,0+j,1,1,0) = 0;
			A (1+j,1+j,1,1,0) = 0;
			A (1+j,2+j,1,1,0) = 0;
			A (1+j,3+j,1,1,0) = x1;
			A (1+j,4+j,1,1,0) = y1;
			A (1+j,5+j,1,1,0) = 1;
			A (1+j,6+j,1,1,0) = -(x1*y2);
			A (1+j,7+j,1,1,0) = -(y1*y2*/
			
			
			/*A.atXY (0+2*j,0) = x1;
			cout << "testin : "<< A.atXY (0,0) << endl;
			A.atXY (0+2*j,1) = y1;
			A.atXY (0+2*j,2) = 1;
			A.atXY (0+2*j,3) = 0;
			A.atXY (0+2*j,4) = 0;
			A.atXY (0+2*j,5) = 0;
			A.atXY (0+2*j,6) = -(x1*x2);
			A.atXY (0+2*j,7) = -(y1*x2);
			cout << "testin : "<< A.atXY (0,7) << endl;
			
			A.atXY (1+2*j,0) = 0;
			A.atXY (1+2*j,1) = 0;
			A.atXY (1+2*j,2) = 0;
			A.atXY (1+2*j,3) = x1;
			A.atXY (1+2*j,4) = y1;
			A.atXY (1+2*j,5) = 1;
			A.atXY (1+2*j,6) = -(x1*y2);
			A.atXY (1+2*j,7) = -(y1*y2);*/
			
			A.atXY (0,0+2*j) = x1;
			//cout << "testin : "<< A.atXY (0,0) << endl;
			A.atXY (1,0+2*j) = y1;
			A.atXY (2,0+2*j) = 1;
			A.atXY (3,0+2*j) = 0;
			A.atXY (4,0+2*j) = 0;
			A.atXY (5,0+2*j) = 0;
			A.atXY (6,0+2*j) = -(x1*x2);
			A.atXY (7,0+2*j) = -(y1*x2);
			//cout << "testin : "<< A.atXY (0,7) << endl;
			
			A.atXY (0,1+2*j) = 0;
			A.atXY (1,1+2*j) = 0;
			A.atXY (2,1+2*j) = 0;
			A.atXY (3,1+2*j) = x1;
			A.atXY (4,1+2*j) = y1;
			A.atXY (5,1+2*j) = 1;
			A.atXY (6,1+2*j) = -(x1*y2);
			A.atXY (7,1+2*j) = -(y1*y2);
			
			
			
			// B
			B.atXY (0,0+2*j) = x2;
			B.atXY (0,1+2*j) = y2;
			

			//current = matches.at(randNum);
			// x=" << descriptors[i].col << " y=" << descriptors[i].row
		}
		//A.invert(true);
		//cout << "test : "<< A.atXY (0,0) << endl;
		//CImg <double> Tr(B.solve(A));
		Tr.assign(B.solve(A));
		//Tr.assign(A.solve(B));
		//A.get_normalize(0,255).save("invert.png");
		//CImg.
	//}
	//cout << "Tr matrix" << endl;
	cout << double(Tr.atXY(0,0)) << "  " << Tr.atXY(0,1) << "  " << Tr.atXY(0,2) << endl;
	cout << Tr.atXY(0,3) << "  " << Tr.atXY(0,4) << "  " << Tr.atXY(0,5) << endl;
	cout << Tr.atXY(0,6) << "  " << Tr.atXY(0,7) << "  " << 1 << endl;
	
	return Tr;
}

    //inverse warping function
 //define the inverse warping function 
CImg<double> inverse_warp ( CImg<double> input)
{
//cout << "1" << endl;	
int w=input._width ;
//cout << "2" << endl;
int	l=input._height ;
//cout << "3" << endl;
CImg<double> output(w,l,1,3,0);
//cout << "4" << endl;
//cout<<output ;
//int i,j,k,m ;
double h[3][3];
h[0][0] = 1.12 ;
h[0][1] = -0.31 ;
h[0][2] = 223 ;
h[1][0] = 0.11 ;
h[1][1] = 0.69 ;
h[1][2] = -19.92 ;
h[2][0] = 0.00026 ;
h[2][1] = -0.000597 ;
h[2][2] = 1 ;

double calc_x =0;
double calc_y =0;
double calc_z =1;
for (int i = 0; i< w; i ++)
{
    for (int j = 0; j< l ; j++)
    {
        calc_x = h[0][0]* i + h[0][1]* j + h[0][2] * 1;
        calc_y = h[1][0]* i + h[1][1]* j + h[1][2] * 1;
        calc_z = h[2][0]* i + h[2][1]* j + h[2][2] * 1; 
        calc_x = calc_x / calc_z ;
        calc_y = calc_y / calc_z ;
        
        // Round of x and y since nearest neighbour will also mean minimum eucledian distance
        calc_x = round (calc_x);
        calc_y = round (calc_y);
        
        //Now get the corresponding value from the input image
        
        if(calc_x >= 0 && calc_y >= 0 && calc_x < w && calc_y < l)
        {
           // output.atXY(i,j) = input.atXY(calc_x, calc_y);
            output.atXY(i,j,0,0) = input.atXY(calc_x, calc_y,0,0);
            output.atXY(i,j,0,1) = input.atXY(calc_x, calc_y,0,1);
            output.atXY(i,j,0,2) = input.atXY(calc_x, calc_y,0,2);
            
        }
        else
        {
            //output.atXY(i,j) = 255 ;
            output.atXY(i,j,0,0) = 255 ;
            output.atXY(i,j,0,1) = 255 ;
            output.atXY(i,j,0,2) = 255 ;
        }
        
    }
}
	
//cout << "5" << endl;
return output ; 

}

CImg<double> inverse_warp ( CImg<double> input, CImg<double> warpingMatrix)
{
//cout << "1" << endl;	
int w=input._width ;
//cout << "2" << endl;
int	l=input._height ;
//cout << "3" << endl;
CImg<double> output(w,l,1,3,0);
////cout << "4" << endl;
////cout<<output ;
////int i,j,k,m ;
double h[3][3];
h[0][0] = warpingMatrix.atXY(0,0) ;
h[0][1] = warpingMatrix.atXY(0,1) ;
h[0][2] = warpingMatrix.atXY(0,2) ;
h[1][0] = warpingMatrix.atXY(1,0) ;
h[1][1] = warpingMatrix.atXY(1,1);
h[1][2] = warpingMatrix.atXY(1,2) ;
h[2][0] = warpingMatrix.atXY(2,0) ;
h[2][1] = warpingMatrix.atXY(2,1) ;
h[2][2] = warpingMatrix.atXY(2,2) ;

double calc_x =0;
double calc_y =0;
double calc_z =1;
for (int i = 0; i< w; i ++)
{
    for (int j = 0; j< l ; j++)
    {
        calc_x = h[0][0]* i + h[0][1]* j + h[0][2] * 1;
        calc_y = h[1][0]* i + h[1][1]* j + h[1][2] * 1;
        calc_z = h[2][0]* i + h[2][1]* j + h[2][2] * 1; 
        calc_x = calc_x / calc_z ;
        calc_y = calc_y / calc_z ;
        
        // Round of x and y since nearest neighbour will also mean minimum eucledian distance
        calc_x = round (calc_x);
        calc_y = round (calc_y);
        
        //Now get the corresponding value from the input image
        
        if(calc_x >= 0 && calc_y >= 0 && calc_x < w && calc_y < l)
        {
           // output.atXY(i,j) = input.atXY(calc_x, calc_y);
            output.atXY(i,j,0,0) = input.atXY(calc_x, calc_y,0,0);
            output.atXY(i,j,0,1) = input.atXY(calc_x, calc_y,0,1);
            output.atXY(i,j,0,2) = input.atXY(calc_x, calc_y,0,2);
            
        }
        else
        {
            //output.atXY(i,j) = 255 ;
            output.atXY(i,j,0,0) = 255 ;
            output.atXY(i,j,0,1) = 255 ;
            output.atXY(i,j,0,2) = 255 ;
        }
        
    }
}
	

return output ; 

}

int getVote (CImg <double> transformationmatrix, vector<SiftDescriptor> descriptors,vector<SiftDescriptor> cmpdescriptors,vector<pair<int,int> > matches)
{
	//inverse_warp 
	CImg <double> ip (1,3);
	CImg <double> op (1,3);
	CImg <double> tr (1,3);
	double x1,x2,y1,y2;
	int votes = 0;
	for (int i =0 ; i< matches.size(); i++)
	{
		x1 = descriptors[matches[i].first].col;
		y1 = descriptors[matches[i].first].row;
		x2 = cmpdescriptors[matches[i].second].col;
		y2 = cmpdescriptors[matches[i].second].row;
		
		ip.atXY(0,0) = x1;
		ip.atXY(0,1) = y1;
		ip.atXY(0,2) = 1;
		
		op.atXY(0,0) = x2;
		op.atXY(0,1) = y2;
		op.atXY(0,2) = 1;
		
		//tr.assign(inverse_warp(ip,transformationmatrix));
//		tr.assign(inverse_warp(op));
//		if (((tr.atXY(0,0)/tr.atXY(0,2))== x1) && ((tr.atXY(0,1)/tr.atXY(0,2))== y1))
//		{
//			votes++;
//		}
         double h[3][3];
        h[0][0] = transformationmatrix.atXY(0,0) ;
        h[0][1] = transformationmatrix.atXY(0,1) ;
        h[0][2] = transformationmatrix.atXY(0,2) ;
        h[1][0] = transformationmatrix.atXY(1,0) ;
        h[1][1] = transformationmatrix.atXY(1,1);
        h[1][2] = transformationmatrix.atXY(1,2) ;
        h[2][0] = transformationmatrix.atXY(2,0) ;
        h[2][1] = transformationmatrix.atXY(2,1) ;
        h[2][2] = transformationmatrix.atXY(2,2) ;

        double calc_x =0;
        double calc_y =0;
        double calc_z =1;

        calc_x = h[0][0]* x2 + h[0][1]* y2 + h[0][2] * 1;
        calc_y = h[1][0]* x2 + h[1][1]* y2 + h[1][2] * 1;
        calc_z = h[2][0]* x2 + h[2][1]* y2 + h[2][2] * 1; 
        calc_x = calc_x / calc_z ;
        calc_y = calc_y / calc_z ;
        
        // Round of x and y since nearest neighbour will also mean minimum eucledian distance
        calc_x = round (calc_x);
        calc_y = round (calc_y);
        
        if (calc_x== x1 && calc_y== y1)
		{
			votes++;
		}
        
 
	}	
	return votes;
}

CImg <double> ransac  (vector<pair<int,int> > matches,vector<SiftDescriptor> descriptors,vector<SiftDescriptor> cmpdescriptors)
{
	int repititions = 100;
	int votes[repititions];
	int maxVote = 0;
	int maxVoteIndex = 0;
	CImg <double> transformationmatrix [repititions] ;
	for (int i = 0 ; i < repititions ; i ++)
	{
            CImg <double> temp = findTransformationMatrix(matches,descriptors, cmpdescriptors);
		transformationmatrix[i].assign(temp);
		votes[i] = getVote (transformationmatrix[i], descriptors, cmpdescriptors, matches);
	}
	
	for (int i = 0 ; i < repititions ; i ++)
	{
		if (maxVote < votes[i])
		{
		    maxVote = votes[i];
		    maxVoteIndex = i ;
		}
	}
	
	return transformationmatrix[maxVoteIndex];
} 


    

    //warping function
    // define the inverse warping function 



    int main(int argc, char **argv) {
        try {

            if (argc < 2) {
                cout << "Insufficent number of arguments; correct usage:" << endl;
                cout << "    a2-p1 part_id ..." << endl;
                return -1;
            }
            //storing the command parameters which are filenames
            std::vector<std::string> fileNames;
            string part = argv[1];
            string inputFile1 = argv[2];
            int noOfImages=argc-3;
            for (int i = 0; i+2 < argc-1; i++) {
                fileNames.push_back(argv[2+i]);
                cout<<"filename"<<fileNames[i]<<" ";
            }
            string inputFile2 = argv[3];
            if (part == "part1") {
                // This is just a bit of sample code to get you started, to
                // show how to use the SIFT library.

                CImg<double> input_image1(inputFile1.c_str());
                CImg<double> input_image2(inputFile2.c_str());


                // convert image to grayscale
                CImg<double> gray1 = input_image1.get_RGBtoHSI().get_channel(2);
                CImg<double> gray2 = input_image2.get_RGBtoHSI().get_channel(2);
                vector<SiftDescriptor> descriptors1 = Sift::compute_sift(gray1);
                vector<SiftDescriptor> descriptors2 = Sift::compute_sift(gray2);

                CImg<double> input_image3(input_image1);
                input_image3.append(input_image2);
                // if arguments are exactly 4
                if (argc == 4)
                    drawSingleImage(descriptors1, descriptors2, input_image3, input_image1, input_image2);

                if (argc > 4) {
                //    sortImages(fileNames, input_image1, descriptors1);
                //    retrievalAlgorithm(fileNames, inputFile1, descriptors1);

                    ExactApprox(fileNames, inputFile1, descriptors1);
                }
                for (int i = 0; i < descriptors1.size(); i++) {
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

                }
            } else if (part == "part2") {
                //call the inverse warping function
                CImg<double> input_image(inputFile1.c_str());
                CImg<double> lincoln_warped = inverse_warp(input_image);
                lincoln_warped.save("lincoln_warped.png");
                
//                // PART 2.2
	    int noOfImages = argc - 3;
	string img_array [noOfImages];
	
	for (int i =0; i<noOfImages ; i++)
	{
		img_array[i] = argv[i + 3];
		//cout << img_array[i];
	}
	    
	  //  CImg<double> input_image(inputFile1.c_str());

		// convert image to grayscale
		CImg<double> gray = input_image.get_RGBtoHSI().get_channel(2);
			vector<SiftDescriptor> descriptors = Sift::compute_sift(gray);
	
	

		for(int i=0; i<descriptors.size(); i++)
		  {
			for(int j=0; j<5; j++)
			  for(int k=0; k<5; k++)
			if(j==2 || k==2)
				for(int p=0; p<3; p++)
						if(descriptors[i].col+k < input_image.width() && descriptors[i].row+j < input_image.height())
						  input_image(descriptors[i].col+k, descriptors[i].row+j, 0, p)=0;

		  }
		  
		  
		  CImg<double> cmp_images[noOfImages];

		
		CImg<double> cmpgray[noOfImages];
			vector<SiftDescriptor> cmpdescriptors[noOfImages];
			
		string savename;
		std::stringstream ss;
		  for (int im =0; im<noOfImages ; im++)
		{
			ss.str(std::string());
			
			cmp_images[im].assign(img_array[im].c_str());
		// convert image to grayscale
			cmpgray[im] = cmp_images[im].get_RGBtoHSI().get_channel(2);
			cmpdescriptors[im] = Sift::compute_sift(cmpgray[im]);
			
			for(int i=0; i<cmpdescriptors[im].size(); i++)
			{
			  for(int j=0; j<5; j++)
			  for(int k=0; k<5; k++)
			if(j==2 || k==2)
				for(int p=0; p<3; p++)
						if(cmpdescriptors[im][i].col+k < cmp_images[im].width() && cmpdescriptors[im][i].row+j < cmp_images[im].height())
						  cmp_images[im](cmpdescriptors[im][i].col+k, cmpdescriptors[im][i].row+j, 0, p)=0;
			}
		  
		  
		}
		for(int i=0; i<cmpdescriptors[0].size(); i++)
		{
			for(int j=0; j<5; j++)
			  for(int k=0; k<5; k++)
				if(j==2 || k==2)
					for(int p=0; p<3; p++)
						if(cmpdescriptors[0][i].col+k < cmp_images[0].width() && cmpdescriptors[0][i].row+j < cmp_images[0].height())
						  cmp_images[0](cmpdescriptors[0][i].col+k, cmpdescriptors[0][i].row+j, 0, p)=0;
			
			
			
		}
		
		for (int im =0; im<noOfImages ; im++)
		{
		    vector<pair<int,int> > matches = allFeatureDistance(descriptors,cmpdescriptors[0]);
		    CImg<double> input_image3(input_image.append(cmp_images[0]));
		    //CImg<double> input_image10 =input_image.invert(false);
		    //input_image10.get_normalize(0,255).save("invert.png");
		    CImg<double> warpingMatrix  = findTransformationMatrix( matches, descriptors,cmpdescriptors[0]);
                    for (int i = 0; i<3 ; i++){
                        cout << warpingMatrix.atXY(i,0) << "  " << warpingMatrix.atXY(i,1) << "  " << warpingMatrix.atXY(i,2) << endl;
                    }
		    CImg<double> warpedImage =  inverse_warp(cmp_images[0],warpingMatrix);
		    //CImg<double> warpedImage =  inverse_warp(cmp_images[0]);
		    //Saving warped image
		    ss << "img_" << (im  + 1)<< "-warped.png";
		    savename = ss.str();
		    //cout << savename << endl;
		    warpedImage.save(savename.c_str());
		    
		}			
		
		

		
	

            } else
                throw std::string("unknown part!");

            // feel free to add more conditions for other parts (e.g. more specific)
            //  parts, for debugging, etc.
        } catch (const string &err) {
            cerr << "Error: " << err << endl;
        }
    }









