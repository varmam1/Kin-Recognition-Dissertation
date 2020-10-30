# Kin Recognition using Weighted Graph-Embedding Based Metric Learning

Core Components Left to Implement
=================================

1. Face detection and saving the faces as an image. 
2. Creating the 4 face descriptors: LBP, HOG, SIFT, VGG
3. Creating the penalty graphs and intrinsic graph
4. Using the graphs to calculate the necessary metrics to determine if the pair of faces is of the given relationship

Timeline and Milestones
=======================

1.  **Oct 23 - Nov 06**:\
    This time will be mainly used for research and getting familiar with
    everything I need to know about the subject such as:

    -   Reading up more in depth on existing literature

    -   Getting a more in depth idea of what architecture I should be
        using

    -   Figuring out which libraries are necessary to be used

    -   Getting an understanding of the libraries to be used by doing
        small quick projects in either Tensorflow or PyTorch.


2.  **Nov 06 - Nov 20**:\
    Basic face recognition will be implemented so that generic images
    can be given and not just images of a specific size of just
    their face.

    -   Using Dlib or OpenCV, create basic facial recognition software
        that outputs the image of the person’s face in the
        required format.

    **Deliverable**: A basic, usable facial recognition model that can
    be used for the rest of the project.

3.  **Nov 20 - Dec 04**:\
    Create the ability to get the necessary feature vectors like local
    binary patterns and the histogram of gradients.

    -   Given the input of the pictures of the faces, get the local
        binary patterns and histogram of gradients out from them.

4.  **Dec 04 - Dec 18**:\
    The VGG-Face CNN descriptors and SIFT face descriptors will be
    obtained in this time which would then be fed into the graphs. If
    this is done before the sprint is up, work will be started on
    implementing the graphs.\
    **Deliverable**: The face descriptor methods are all created.

5.  **Dec 18 - Jan 01**:\
    The implementation of the intrinsic graph would be created in
    this sprint. This should create the graph based on the
    class information.

6.  **Jan 01 - Jan 15**:\
    The penalty graphs should be created in this sprint and thus get the
    calculations necessary to figure out the kin relationship. This
    overall model that is created should be able to fulfill the success
    criterion.\
    **Deliverable**: A model that can predict kin relationships between
    pairs of images.

7.  **Jan 15 - Jan 29**:\
    At this point, ablation studies will be done to help to evaluate the
    network by seeing which inputs are necessary for the network to work
    with more focus being given to the progress report and presentation:

    -   Start work on an ablation study.

    -   Write up the Progress Report and create a presentation for it to
        be handed in on February 5th.

    **Deliverable**: Progress Report

8.  **Jan 29 - Feb 12**:\
    Finish work on the ablation studies and start work on the first
    extension:

    -   Finish up ablation studies for evaluation and finish model

    -   Start working on extending the model to work for videos as well.

    **Deliverable**: A finalized model that reports the kin relation
    between a people in a pair of images.

9.  **Feb 12 - Feb 26**:\
    Finish work on the first extension and start doing work on a second
    extension to try and improve on the model for images.

    **Deliverable**: A new model which is the model implemented which is
    extended to work on videos

10. **Feb 26 - Mar 12**:\
    At this point, the first draft of the dissertation starts to be
    written alongside some work on improving the model, with priority
    given to the draft:

    -   Write out a draft of the Introduction and Preparation Chapters

    -   Work on second extension for improving the existing model.

11. **Mar 12 - Mar 26**:\
    Continue working on writing the dissertation and extension with the
    same priority:

    -   Write out a draft of the Implementation chapter

    -   Work on second extension for improving the existing model. If it
        is finished, then this will be included in the
        implementation chapter. If not, it will be abandoned in favor of
        finishing up the draft of the dissertation.

    **Deliverable**: A draft of the Introduction, Preparation and
    Implementation chapters to be given to supervisor for feedback

12. **Mar 26 - Apr 09**:\
    Continue working on writing the dissertation:

    -   Write out a draft of the Evaluation chapter

    -   Revise the Introduction, Preparation and Implementation chapter
        based on supervisor feedback

13. **Apr 09 - Apr 23**:\
    Continue working on writing the dissertation:

    -   Write out a draft of the Conclusion chapter

    **Deliverable**: A draft dissertation to be given to supervisor for
    feedback

14. **Apr 23 - May 14**:\
    If everything has gone well, this should be where final touches are
    applied. **Deliverable**: The completed dissertation
