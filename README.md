# Virtual Reality Modeling project (part 1)

Facial emotions dataset used in this project can be found at: <a>https://github.com/muxspace/facial_expressions</a>

This repository is a subpart of a project "vr-facial-expressions" for the Kaunas University of Technology VR-Modeling module.

The second part can be found here: <a href="https://github.com/gMatas/vr-facial-expressions-quiz">vr-facial-expressions-quiz</a>

This facial emotions classifier was specifically made only using OpenCV library. The SVM classifier was trained only on 4 emotions classes using just ~300 labeled images for each of the class, due to lack of good quality samples in the used dataset (even though it was the best public dataset for this task that I managed to find).

Trained emotions and their labels:
<ul>
  <li>Anger - 0</li>
  <li>Happiness/joy - 3</li>
  <li>Surprise - 5</li>
  <li>Neutral - 6</li>
</ul>

The overall classifier accuracy (tested on total of 127 images) is 73%
