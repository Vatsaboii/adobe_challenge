# adobe_challenge
This repository will contain the code for the Adobe Challenge we received. It will serve as a central location for all related scripts, documentation, and updates throughout the project.

**Description about the Project**

The project involved enhancing image processing by refining and completing curves within test case images. The goal was to smooth, join, and regularize irregular or broken curves, ensuring a more accurate and consistent output. The project also focused on removing extraneous elements outside defined boundaries, ultimately improving the precision and reliability of the image processing model. This was a crucial step in refining our model’s ability to handle complex input data effectively.

**How we dealt with the problem statement?**
- Initial Image Data Handling: Developed a script to convert CSV data of images into JPG format, enabling smooth and efficient processing by the model.
- Shape Detection and Annotation: Implemented detection algorithms to identify and annotate various shapes within the uploaded images.
- Bezier Curves Application: Used Bezier curves to smooth and regularize the detected shapes, ensuring more accurate and visually appealing results.
- Regularization of Shapes: Enhanced the shapes by applying regularization techniques, smoothing irregular curves, and correcting bent or distorted outlines.
- Symmetry Analysis: Analyzed the symmetry of the detected shapes, ensuring that the regularized shapes maintained symmetrical properties where applicable.
- Handling Incomplete Curves: Developed methods to detect and complete incomplete curves, using Bezier curves to fill gaps and create smooth transitions. Added functionality to cut through other curves if necessary, to ensure the completion of broken or unfinished shapes.
- Occlusion Handling: Implemented techniques to manage occlusion, ensuring that incomplete curves or shapes are properly completed even if parts are obscured.
- Boundary Handling: Modified the processing pipeline to ignore and remove lines or curves detected outside the square image boundaries, keeping the rest of the image intact.
- Comparison and Visualization: Visualized the differences between the original and processed images, highlighting where curves were completed, joined, or removed.
- Final Testing and Refinement: Conducted thorough testing with various test case images to validate the accuracy of the model.
Refined the algorithms based on test results to improve performance and reliability.

**Description of the Files present in the Repository**
1. a1.py -> This file was originally created to handle the detection and annotation of shapes identified in the uploaded images.
2. csvtojpg.py -> This file was developed to convert CSV data of images into JPG format, enabling smooth and efficient model execution.
3. merge.py -> This file handles shape detection and annotation, regularization, symmetry analysis, occlusion of incomplete curves, and image completion.
4. merged.py -> Everything is similar as in merge.py except for the fact that this code does not annotate the shapes present in the image.
5. Test_cases_images -> This folder contains the test images used to run the model.
6. Test_cases_solutions -> This folder contains the output images afte the test case images are run in the model.

