# 11-785 Project - Neural Day to Night

Team members:  
Chi Liu (chil1)  
Raphael Olivier (rolivier)  
Teven Le Scao (tlescao)  
Jean-Baptiste Lamare (jlamare)

# Abstruct
The photography and cinema industries rely heavily on techniques to turn day pictures into night, called day-to night tricks. These techniques usually required special shooting conditions, and/or an image editing software that could not work without some manual intervention. The recent improvement of machine learning based computer vision techniques has allowed the development of algorithms that could turn day into night automatically.

Given an outdoor picture, our objective is to change the perceived time of the day of the picture. Many movies, such as Mad Max : Fury Road (2015)  use such techniques in order to shoot with the optimal lighting conditions of day-time but still achieve the desred night-time mood.
In this report we focus on turning day into night, although a similar pipeline can be applied to the opposite translation, or many more.


# Poster
![1](/cgan_result.png)

[[Watch the Result Video]](https://youtu.be/pEj0Ksczb3I)

# Out Of Domain Generation
In order to check how robust our model is and to highlight what the model is learning, we applied it on an out-of-domain picture of the CMU campus. The result is reported in  the following figure. The model tries to identify typical elements of the training images : it recognizes trees well, but tries to light the background up without a good reason here, as the dataset images typically feature streelights in the distance, with heavy reflections on the rain of the windshield. An interesting point is how all of the finer details get painted over with very bright colours : indeed, in the training data, those tend to be objects that light up at night, like streetlights, car lights, or bus stops. The result could not fool human perception, but still learns structures at different scales.

![2](/cmu_source.png)

![3](/cmu_target.png)


# Conclusion
We applied conditional GANs to the day-night image-to-image translation problem, on a more challenging dataset than is usual for the task. We have explored different architectures and performed error analysis, showing the influence and limits of tricks such as skip connections in CNNs and L1 regularization in hard generation tasks. We also explored an original way to combine models in generation tasks, by training a generator with several discriminators simultaneously, and showed that this method can yield improvements over simpler models. Although the quality of the generated images at this time is still far from movie editing standards, we hope that this work and analysis can contribute to future progress in image translation tasks.
