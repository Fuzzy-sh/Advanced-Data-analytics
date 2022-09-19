<a href="https://www.linkedin.com/in/fuzzy-shahidi"><img src="https://img.shields.io/badge/Linkdin-Fuzzy%20Shahidi-blue.svg" alt="DOI"></a>



# Advanced-Data-analytics

## Aim and motivation

<pre>
If we can predict homeless people’s behaviour, we will able to provide help and services with these people. 
Among all methods, machine learning techniques have been proved that they are able to improve the decision making in the health-care sector (Chen et al., 2019). 
Session-based recommenders  are useful when we have user interaction history  that they can learn based on the short-term interaction (Wang et al., 2022). These methods are emerging in the healthcare system to recommend the next-treatment recommendation (Haas, n.d.). 
Our Aim is to predict the event within a session.  
We used Word2vec model (Rong, 2016) that that capture the semantic similarities to predict the next event.

</pre>

## Data set

<pre>
* In this work we used the MLB public dataset to represent the medical data. 
* The features in this dataset are correlated with the features that we will see in the real dataset. That’s why this dataset represents health care dataset.
* The data contains, a series of discreet events, including medical tests that can come back with good or bad results or vital crash that needs emergency or intense medical aid. 
* Another type of events in our database are stretched over a period. These events have starting and ending point 
</pre>


## Preprocessing Method

![image](https://user-images.githubusercontent.com/38839459/191089896-d2205df9-f049-4e74-bc83-47937704894e.png)


![image](https://user-images.githubusercontent.com/38839459/191089929-5d9e7ce3-2825-409c-8c3c-cf475ec8e6a8.png)

## To deal with imbalanced Classes, we used Weighted Random Sampler:
<img width="595" alt="image" src="https://user-images.githubusercontent.com/38839459/191090177-efa4060d-0c39-4ce9-aa14-75a33ad5d2bf.png">

![image](https://user-images.githubusercontent.com/38839459/191090189-21653a44-29a3-44f7-a3f4-3da744a32a97.png)


## Machine learning Method
![image](https://user-images.githubusercontent.com/38839459/191090263-d6c7a20b-8980-4978-86cf-5cd8c280a238.png)


## Results and cluster performance
<pre>

By using job arrays and creating a loop in the shell
* Submitted several jobs on the GPU partition. 
* Each job had unique input to do hyper parameter optimization
* We have successfully received the results for about  200 jobs

</pre>

<img width="278" alt="image" src="https://user-images.githubusercontent.com/38839459/191090854-be5ac47e-826d-4a54-9dba-f11c17888bac.png">

<img width="225" alt="image" src="https://user-images.githubusercontent.com/38839459/191090904-ac2b1bfa-b2b8-4476-9b15-82ea3ccd7700.png">


## Conclusion and Reflection

<pre>

* Submitted several jobs on the GPU partition. 
* In each job I trained the model with 1500 epoch
* Each job took about 40 mins on GPU  (about 6 hours on CPU partition) 
* By observing the jobs on the cluster each time 12 jobs was running in parallel.
* Whole of the experiments took about 5 days in the cluster which is almost equal to the 60 days in personal laptop.
* Could use the cluster to find the best parameters almost 10 times faster than using my own resource.
* Found clusters recourses very useful and and they are time saving for doing experiments 



</pre>


