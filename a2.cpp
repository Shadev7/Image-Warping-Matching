
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

float comparingvalues(CImg<double> input_image3, CImg<double> input_image1, vector<SiftDescriptor> &descriptorsip, vector<SiftDescriptor> &descriptors, int taskno, int bestin) {
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
    int val = comparingvalues(input_image3, input_image1, descriptors1, descriptors2, 1, -1);
}


//generating gaussian vector

int findGaussian(string queryimgimg, string fileName, vector<SiftDescriptor> &descriptorsip, vector<SiftDescriptor> &descriptors, CImg<double> queryimg) {
    vector< vector<double> > dim_query_matrix(descriptorsip.size(), vector<double> (10));
    vector< vector<double> > dim_sample_matrix(descriptors.size(), vector<double> (10));
    vector<vector<double> > values(10, vector<double> (128));
    float best, secondbest, bestreal, secndbestreal;
    int bestindex, secondindex, bestindreal, secbestindexreal;
    int count = 0, ct = 0;
    //creating the gaussian 128D random matrix of dimension 10
    for (int k = 0; k < 10; k++) {
        std::vector<int> descriptor;
        //generating the gaussian random vectors
        for (int n = 0; n < 128; n++) {
            double a = ((double) rand() / (RAND_MAX + 1.0));
            values[k][n] = a;
        }
    }
    //finding the dot product of gaussian vector and descriptors of query image along dimensions
    for (int i = 0; i < descriptorsip.size(); i++) {
        for (int j = 0; j < 10; j++) {
            int sum = 0;
            for (int k = 0; k < 128; k++) {
                sum = sum + descriptorsip[i].descriptor[k] * values[j][k];
            }
            int dotpdt = sum / 200;
            dim_query_matrix[i][j] = dotpdt;
        }
    }
    //finding the dot product of gaussian vector and descriptors of example image along dimensions
    for (int i = 0; i < descriptors.size(); i++) {
        for (int j = 0; j < 10; j++) {
            int sum = 0;
            for (int k = 0; k < 128; k++) {
                sum = sum + descriptors[i].descriptor[k] * values[j][k];
            }
            int dotpdt = sum / 200;
            dim_sample_matrix[i][j] = dotpdt;
        }
    }
    //finding the closest summary vectors
    vector< int> bestimgindexes;
    vector< int> bestsmplindexes;
    int l = 0;
    //comparing for nearest neighbours in k dimensions
    for (int i = 0; i < dim_query_matrix.size(); i++) {
        for (int j = 0; j < dim_sample_matrix.size(); j++) {
            int sum = 0;
            for (int k = 0; k < 10; k++) {
                float diff = (dim_query_matrix[i][k] - dim_sample_matrix[j][k]);
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
        if (best / secondbest < 0.70) {
            bestimgindexes.push_back(i);
            bestsmplindexes.push_back(bestindex);
        }
    }
    //using this subset of nearest neighbors find appropriate SIFT vectors in the 128 dimension
    for (int i = 0; i < bestimgindexes.size(); i++) {
        for (int j = 0; j < bestimgindexes.size(); j++) {
            int sum = 0;
            for (int m = 0; m < 128; m++) {
                float diff = (descriptorsip[bestimgindexes[i]].descriptor[m])-(descriptors[bestsmplindexes[j]].descriptor[m]);
                sum = sum + (diff * diff);
            }
            float dist = sqrt(sum);
            //keep initial distance as best
            if (j == 0) {
                best = dist;
                secondbest = dist;
                bestindex = j;
                secondindex = j;
            } else {
                if (dist < secondbest && dist > best) {
                    secondbest = dist;
                    secondindex = j;

                } else if (dist < best) {
                    best = dist;
                    bestindex = j;
                } else
                    continue;
            }
        }
        if (best / secondbest < 0.8)
            ct = ct + 1;
    }
    return ct;
}

//Finds inverse of 3x3 matrix
CImg<double> invertMatrix(double h[3][3])
{
	CImg<double> h_inverse (3,3);
	
	double mod_h = h[0][0]*(h[1][1]*h[2][2] - h[1][2]*h[2][1])- h[0][1] * (h[1][0]*h[2][2] - h[1][2]*h[2][0] ) + h[0][2]*(h[1][0]*h[2][1] - h[1][1]*h[2][0]);
	
	h_inverse.atXY(0,0) = (h[1][1]* h[2][2] - h[1][2] * h[2][1]) / mod_h;	
	h_inverse.atXY(0,1) = (h[0][2]* h[2][1] - h[0][1] * h[2][2]) / mod_h;	
	h_inverse.atXY(0,2) = (h[0][1]* h[1][2] - h[0][2] * h[1][1]) / mod_h;
	h_inverse.atXY(1,0) = (h[1][2]* h[2][0] - h[1][0] * h[2][2]) / mod_h;	
	h_inverse.atXY(1,1) = (h[0][0]* h[2][2] - h[0][2] * h[2][0]) / mod_h;	
	h_inverse.atXY(1,2) = (h[0][2]* h[1][0] - h[0][0] * h[1][2]) / mod_h;
	h_inverse.atXY(2,0) = (h[1][0]* h[2][1] - h[1][1] * h[2][0]) / mod_h;
	h_inverse.atXY(2,1) = (h[0][1]* h[2][0] - h[0][0] * h[2][1]) / mod_h;
	h_inverse.atXY(2,2) = (h[0][0]* h[1][1] - h[0][1] * h[1][0]) / mod_h;
	
	return h_inverse;
}

bool compare(const pair<int, int> &a, const pair<int, int> &b) {
    return a.second < b.second;
}

// finding the precision value for all images that are correctly matched from the same attraction

void retrievalAlgorithm(vector<string> fileNames, string queryimg, vector<SiftDescriptor> &descriptorsip, int menu) {
    std::vector<float> counts;
    std::vector<int> indexes;
    vector<pair<int, int> > values;
    //loop over images to find features
    for (int k = 0; k < fileNames.size(); k++) {
        CImg<double> input_image(fileNames[k].c_str());
        CImg<double> query_image(queryimg.c_str());
        //convert to greyscale
        CImg<double> gray = input_image.get_RGBtoHSI().get_channel(2);
        vector<SiftDescriptor> descriptors = Sift::compute_sift(gray);
        int count = 0;
        if (menu == 2)
            count = comparingvalues(query_image, query_image, descriptorsip, descriptors, 2, -1);
        else
            count = findGaussian(queryimg, fileNames[k], descriptorsip, descriptors, input_image);
        //  store the counts to keep track
        counts.push_back(count);
        indexes.push_back(count);
        pair<int, int> sub;
        sub.first = k;
        sub.second = count;
        values.push_back(sub);

    }
    sort(counts.begin(), counts.end());
    sort(values.begin(), values.end(), compare);
    float precision = 0;
    std::string::size_type pos = queryimg.find('_');
    std::string queryim;
    // display the sorted list of images based on  matched feature 
    for (int i = counts.size() - 1; i >= 0; i--) {
        cout << fileNames[values[i].first] << endl;
    }

    if (pos != std::string::npos)
        queryim = queryimg.substr(0, pos);
    cout << endl << "Top ten images are:" << endl;
    int ctr = 0;
    // display the sorted list of images based on  matched feature 
    for (int i = counts.size() - 1; i >= 0; i--) {
        if (ctr > 10)
            break;
        else {
            //  int pos = find(indexes.begin(), indexes.end(), counts[i]) - indexes.begin();
            // if (pos >= indexes.size())
            //   cout << "error";
            //else {
            cout << fileNames[values[i].first] << endl;
            if (fileNames[values[i].first].find(queryim) != std::string::npos)
                precision = precision + 1;
        }
        ctr = ctr + 1;
    }
    float val = (counts.size() < 10) ? counts.size() : 10.0;
    cout << endl << "precisions is : " << (precision / 10.0) << " after ranking with the top ten images" << endl;
}

vector<pair<int, int> > allFeatureDistance(CImg<double> input_image3, CImg<double> input_image1, vector<SiftDescriptor> &descriptorsip, vector<SiftDescriptor> &descriptors) 
{
    float best, secondbest;
    int bestindex, secondindex;
    vector<pair<int, int> > matches;
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
            pair<int, int> match;
            match.first = i;
            match.second = bestindex;
            matches.push_back(match);
        }
    }
    return matches;

}

CImg <double> findTransformationMatrix(vector<pair<int, int> > matches, vector<SiftDescriptor> descriptors, vector<SiftDescriptor> cmpdescriptors) 
{
    int min = 0;
	int max = matches.size();
	int randNum ;
	int noOfSamples = 4;
	double x1, y1, x2, y2;
	
	vector<vector<double> > ip;
	vector<double> temp;
	vector<double> op;
	CImg <double> A (8,8,1,1);
	
	CImg <double> B (1,8,1,1);
	CImg <double> Tr (1,8,1,1);
	//CImg <double> TxMatrix (3,3,1,1);
	int random_match_ip;
	int random_match_op;


		for (int j = 0; j< noOfSamples ; j++)
		{
			randNum = rand()%(max-min + 1) + min;
			while(randNum >= matches.size())
			{
				randNum = rand()%(max-min + 1) + min;
			}
			random_match_ip = matches[randNum].first;
			random_match_op = matches[randNum].second;
			x1 = descriptors[random_match_ip].col;
			y1 = descriptors[random_match_ip].row;
			
			x2 = cmpdescriptors[random_match_op].col;
			y2 = cmpdescriptors[random_match_op].row;

			//Odd row
			A.atXY (0,0+2*j) = x1;
			A.atXY (1,0+2*j) = y1;
			A.atXY (2,0+2*j) = 1;
			A.atXY (3,0+2*j) = 0;
			A.atXY (4,0+2*j) = 0;
			A.atXY (5,0+2*j) = 0;
			A.atXY (6,0+2*j) = -(x1*x2);
			A.atXY (7,0+2*j) = -(y1*x2);
			
			//Even row
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
			
		}

		Tr.assign(B.solve(A));

	
	/* for (int i = 0; i< 3; i++)
	{
		for (int j = 0; j< 3 ; j++)
		{
			TxMatrix.atXY(i,j) = Tr.atXY(0, i + j); // check i, j
		}
	} */

	
	return Tr;
	

    
}




//inverse warping function
//define the inverse warping function 

CImg<double> inverse_warp ( CImg<double> input)
{

    int w = input._width;

    int l = input._height;

    CImg<double> output(w, l, 1, 3, 0);

    double h[3][3];
    h[0][0] = 0.907;
    h[0][1] = 0.258;
    h[0][2] = -182;
    h[1][0] = -0.153;
    h[1][1] = 1.44;
    h[1][2] = 58;
    h[2][0] = -0.000306;
    h[2][1] = 0.000731;
    h[2][2] = 1;


    CImg<double> hiMatrix(invertMatrix(h));
    double calc_x = 0;
    double calc_y = 0;
    double calc_z = 1;
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < l; j++) {
            calc_x = hiMatrix.atXY(0, 0) * i + hiMatrix.atXY(0, 1) * j + hiMatrix.atXY(0, 2) * 1;
            calc_y = hiMatrix.atXY(1, 0) * i + hiMatrix.atXY(1, 1) * j + hiMatrix.atXY(1, 2) * 1;
            calc_z = hiMatrix.atXY(2, 0) * i + hiMatrix.atXY(2, 1) * j + hiMatrix.atXY(2, 2)* 1;

            calc_x = calc_x / calc_z;
            calc_y = calc_y / calc_z;

            // Round of x and y since nearest neighbour will also mean minimum eucledian distance
            calc_x = round(calc_x);
            calc_y = round(calc_y);

            //Now get the corresponding value from the input image

            if (calc_x >= 0 && calc_y >= 0 && calc_x < w && calc_y < l) {
                //For RGB planes
                output.atXY(i, j, 0, 0) = input.atXY(calc_x, calc_y, 0, 0);
                output.atXY(i, j, 0, 1) = input.atXY(calc_x, calc_y, 0, 1);
                output.atXY(i, j, 0, 2) = input.atXY(calc_x, calc_y, 0, 2);

            } else {
                //For RGB planes
                output.atXY(i, j, 0, 0) = 255;
                output.atXY(i, j, 0, 1) = 255;
                output.atXY(i, j, 0, 2) = 255;
            }

        }
    }


    return output;

}

CImg<double> inverse_warp ( CImg<double> input, CImg<double> warpingMatrix)
{
    int w = input._width;
    int l = input._height;
    CImg<double> output(w, l, 1, 3, 0);


    CImg<double> hiMatrix(3, 3);
    hiMatrix.assign(warpingMatrix);
    double calc_x = 0;
    double calc_y = 0;
    double calc_z = 1;
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < l; j++) {
            calc_x = hiMatrix.atXY(0, 0) * i + hiMatrix.atXY(0, 1) * j + hiMatrix.atXY(0, 2) * 1;
            calc_y = hiMatrix.atXY(1, 0) * i + hiMatrix.atXY(1, 1) * j + hiMatrix.atXY(1, 2) * 1;
            calc_z = hiMatrix.atXY(2, 0) * i + hiMatrix.atXY(2, 1) * j + hiMatrix.atXY(2, 2)* 1;

            calc_x = calc_x / calc_z;
            calc_y = calc_y / calc_z;

            // Round of x and y since nearest neighbour will also mean minimum eucledian distance
            calc_x = round(calc_x);
            calc_y = round(calc_y);

            //Now get the corresponding value from the input image

            if (calc_x >= 0 && calc_y >= 0 && calc_x < w && calc_y < l) {
                output.atXY(i, j, 0, 0) = input.atXY(calc_x, calc_y, 0, 0);
                output.atXY(i, j, 0, 1) = input.atXY(calc_x, calc_y, 0, 1);
                output.atXY(i, j, 0, 2) = input.atXY(calc_x, calc_y, 0, 2);

            } else {
                output.atXY(i, j, 0, 0) = 255;
                output.atXY(i, j, 0, 1) = 255;
                output.atXY(i, j, 0, 2) = 255;
            }

        }
    }


    return output;

}

int getVote (CImg <double> transformationmatrix, vector<SiftDescriptor> descriptors,vector<SiftDescriptor> cmpdescriptors,vector<pair<int,int> > matches)
{
	//inverse_warp 
	CImg <double> ip (1,3);
	CImg <double> op (1,3);
	CImg <double> tr (1,3);
	double x1,x2,y1,y2;
	double threshold = 0.02;
	int votes = 0;
	for (int i =0 ; i< matches.size(); i++)
	//for (int i =0 ; i< 5; i++)
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
		
         double h[3][3];

		
		h[0][0] = transformationmatrix.atXY(0,0) ;
        h[0][1] = transformationmatrix.atXY(0,1) ;
        h[0][2] = transformationmatrix.atXY(0,2) ;
        h[1][0] = transformationmatrix.atXY(0,3) ;
        h[1][1] = transformationmatrix.atXY(0,4);
        h[1][2] = transformationmatrix.atXY(0,5) ;
        h[2][0] = transformationmatrix.atXY(0,6) ;
        h[2][1] = transformationmatrix.atXY(0,7) ;
        h[2][2] = 1 ;
		

		CImg<double> hiMatrix(invertMatrix(h));
        double calc_x =0;
        double calc_y =0;
        double calc_z =1;

		//Inverse transform
		calc_x = hiMatrix.atXY(0,0)* x2 + hiMatrix.atXY(0,1)* y2 + hiMatrix.atXY(0,2) * 1;
		calc_y = hiMatrix.atXY(1,0)* x2 + hiMatrix.atXY(1,1)* y2 + hiMatrix.atXY(1,2) * 1;
		calc_z = hiMatrix.atXY(2,0)* x2 + hiMatrix.atXY(2,1)* y2 + hiMatrix.atXY(2,2)* 1;
        
		
        // Round of x and y since nearest neighbour will also mean minimum eucledian distance
        calc_x = round (calc_x);
        calc_y = round (calc_y);
		calc_x = calc_x / calc_z ;
        calc_y = calc_y / calc_z ;
		

        //if (calc_x == x1 && calc_y == y1)
		if ((calc_x >= (1 - threshold) * x1 && calc_x <= (1 + threshold) * x1) && (calc_y >= (1 - threshold) * y1 && calc_y <= (1 + threshold) * y1))
		{
			votes++;
		}
 
	}
	return votes;
}



CImg <double> ransac  (vector<pair<int,int> > matches,vector<SiftDescriptor> descriptors,vector<SiftDescriptor> cmpdescriptors)
{
	int repititions = 10000;
	int votes[repititions];
	int maxVote = 0;
	int maxVoteIndex = 0;
	CImg <double> transformationmatrix [repititions] ;

	for (int i = 0 ; i < repititions ; i ++)
	{
        CImg <double> temp (findTransformationMatrix(matches,descriptors, cmpdescriptors));
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
		
	
	CImg <double> tx_Matrix (3,3);

	
	//Convert to a 3x3 matrix
	tx_Matrix.atXY(0,0) = transformationmatrix[maxVoteIndex].atXY(0,0) ;
    tx_Matrix.atXY(0,1) = transformationmatrix[maxVoteIndex].atXY(0,1) ;
    tx_Matrix.atXY(0,2) = transformationmatrix[maxVoteIndex].atXY(0,2) ;
    tx_Matrix.atXY(1,0) = transformationmatrix[maxVoteIndex].atXY(0,3) ;
    tx_Matrix.atXY(1,1) = transformationmatrix[maxVoteIndex].atXY(0,4);
    tx_Matrix.atXY(1,2) = transformationmatrix[maxVoteIndex].atXY(0,5) ;
    tx_Matrix.atXY(2,0) = transformationmatrix[maxVoteIndex].atXY(0,6) ;
    tx_Matrix.atXY(2,1) = transformationmatrix[maxVoteIndex].atXY(0,7) ;
    tx_Matrix.atXY(2,2) = 1 ;

	
	return tx_Matrix;
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
        int noOfImages = argc - 3;
        for (int i = 0; i + 3 < argc; i++) {
            fileNames.push_back(argv[i + 3]);
        }
        string inputFile2 = argv[3];
        CImg<double> input_image1(inputFile1.c_str());
        CImg<double> input_image2(inputFile2.c_str());

        //// convert image to grayscale
        CImg<double> gray1 = input_image1.get_RGBtoHSI().get_channel(2);
        CImg<double> gray2 = input_image2.get_RGBtoHSI().get_channel(2);
        vector<SiftDescriptor> descriptors1 = Sift::compute_sift(gray1);
        vector<SiftDescriptor> descriptors2 = Sift::compute_sift(gray2);

        CImg<double> input_image3(input_image1);
        input_image3.append(input_image2);

        if (part == "part1") {
            //            // if arguments are exactly 4
            if (argc == 4)
                drawSingleImage(descriptors1, descriptors2, input_image3, input_image1, input_image2);

            if (argc > 4) {
                retrievalAlgorithm(fileNames, inputFile1, descriptors1, 2);

            }
        } else if (part == "part1fast") {
            retrievalAlgorithm(fileNames, inputFile1, descriptors1, 4);
        } else if (part == "part2") {

            //call the inverse warping function
            CImg<double> input_image(inputFile1.c_str());
            CImg<double> lincoln_warped = inverse_warp(input_image);
            lincoln_warped.save("lincoln_warped.png");
                
		// PART 2.2
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
		  
			}

			for (int im =0; im<noOfImages ; im++)
			{
				ss.str(std::string());
				//vector<pair<int,int> > matches = allFeatureDistance(descriptors,cmpdescriptors[im]);
				vector<pair<int, int> > matches = allFeatureDistance(input_image, cmp_images[im], descriptors, cmpdescriptors[im]);

				CImg<double> warpingMatrix  (ransac( matches, descriptors,cmpdescriptors[im]));
				CImg<double> warpedImage =  inverse_warp(cmp_images[im],warpingMatrix);
				
				//Saving warped image
				//string fName = 
				ss << "img" << (im  + 1)<< "_warped.png";
				savename = ss.str();
				
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


