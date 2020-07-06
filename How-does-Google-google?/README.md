### Project Overview

 Problem Statement

This assessment will involve finding out the importance of web pages with the help of Google's Pagerank algorithm. In case you don't know, Pagerank is the algorithm employed by Google to sort webpages without human evaluation of the content. You will be solving it with the help of eigen value decomposition. But first understand how Pagerank does sorting of webpages.

PageRank Explained

If you've ever created a web page, you've probably included links to other pages that contain valuable, reliable information. By doing so, you are affirming the importance of the pages you link to. The fundamental idea put forth by PageRank's creators, Sergey Brin and Lawrence Page, is this: the importance of a page is judged by the number of pages linking to it as well as their importance. 

We will assign to each web page P a measure of its importance I(P), called the page's PageRank. Here's how the PageRank is determined. Suppose that page Pj has lj links. If one of those links is to page Pi, then Pj will pass on 1/lj of its importance to Pi . The importance ranking of Pi is then the sum of all the contributions made by pages linking to it. That is, if we denote the set of pages linking to Pi by Bi , then 
 
Creation of Hyperlink matrix

Let's first create a matrix, called the hyperlink matrix, H = [Hij] in which the entry in the i-th row and j-th column is 
Notice that H has some special properties. First, its entries are all nonnegative. Also, the sum of the entries in a column is one unless the page corresponding to that column has no links. 
We will also form a vector I=[I(Pi)] whose components are PageRanks--that is, the importance rankings of all the pages. The condition above defining the PageRank may be expressed as I=HI . In other words, the vector I is an eigenvector of the matrix H with eigenvalue 1. We also call this a stationary vector of H.

Reference: http://www.ams.org/publicoutreach/feature-column/fcarc-pagerank


### Learnings from the project

 Doing this project will help you brush up the following skills:
•	Common vector and matrix operations
•	Vector and matrix products
•	Eigen vector decomposition



### Approach taken to solve the problem

 Calculate the top ranked page with eigenvector decomposition

You will be creating the matrix H and its eigenvector with eigenvalue of 1 to find the importance of webpages.
Instructions
•	The adjacency matrix adj_mat is provided to you. 
•	For this matrix perform the eigen vector decomposition using .linalg.eig() method of numpy. This function returns a tuple and save them as eigenvalues and eigenvectors
•	The following is a single step divided into small steps:
o	Find the eigen vector corresponding to 1 from eigenvectors (first column of eigenvectors), that is abs(eigenvectors[:,0])
o	Normalize this by dividing with np.linalg.norm(eigenvectors[:,0],1). Save it as eigen_1
•	Next save the most important page number by finding the index with highest value within eigen_1. This can be done by using the .where() method from numpy Save it as page and print it out.

Calculate top ranked page with Power Method

The page ranking problem can also be solved by using the power method. So what is it? 
We begin by choosing a vector I0 as a candidate for I and then producing a sequence of vectors Ik by the transformation IK+1 = H.IK . The general principle is that the sequence IK will converge to the stationary vector I.
Steps
•	Initialize a stationary matrix I
•	Over a number of iterations update I by multiplying with H
In this task you will be solving the pagerank problem for the same network but this time with the power method.
Instructions
•	The hyperlink matrix adj_mat is already defined for you. Initialize a stationary matrix init_I which has 1 at the first position and 0s in the rest 7 blocks of the numpy ndarray.
•	Use a for loop over 10 iterations where you update adj_mat according to the rule IK+1 = H.IK, this can be done by .dot(adj_mat, init_I) . Also normalize init_I at every iteration using np.linalg.norm(init_I, 1)
•	Save the page number with highest importance as power_page. This can be found by .where() as done in the previous task.

Problem with Power Method

Instructions
•	The new adjacency matrix this time for the new webpage connection structure shown in the above image. It is provided as new_adj_mat
•	Initialize a stationary matrix new_init_I in the same manner as you did for the previous task
•	Use a for loop to iterate 10 times and update in the similar manner as you did for the previous task i.e. first take dot product np.dot(new_adj_mat, new_init_I) and then normalize as done in previous task.
•	Print out new_init_I to check out its result. Observe how you get pagerank value for 3rd webpage as zero. Is it not possible as it has incoming connections. 

Modified Power Method 

For this task you need to do the modifications to the earlier matrix where the page rank failed and modify it. Use G = αS + (L−α)(1/n)L as the new hyperlink matrix and L is a n∗n matrix whose all entries are 1.   α can take any value between 0 and 1. 
Instructions
•	Initialize new hyperlink matrix G with the help of the mathematical formula given above. In the formula n can be taken as len(new_adj_mat) and 1 as np.ones(new_adj_mat.shape).           Save it as G
•	Initialize stationary vector as final_init_I consisting of 1 at its beginning and rest all zeros in a 1D NumPy array
•	Perform 1000 iterations using a for loop to update the stationary vector in the same manner as for the Power Method. Also, do not forget to normalize it.
•	Print out final_init_I



