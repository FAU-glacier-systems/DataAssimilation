# DataAssimilation

This repository contains the code for a data assimilation strategy coupling the Ensebmle Kalman Filter with the Instructed glacier model [Instructed Glacier Model (IGM)]([URL](https://github.com/jouvetg/igm)).
1. The OGGM-shop module of IGM is used to download glacier related maps. 
2. The inversion module of IGM is used to optimize thicknes and slinding coefficient.
3. Our data assimilation script uses the Ensemble Kalman filter to infer Surface Mass Balance Parameters of the simple SMB module from IGM.

   Observations are generated ether using IGM or from the elevation change map of Hugonnet et al.

![Iterations of Ensemble Kalman Filter](Figures/Rhone_3D.gif)
![Pipeline](Figures/pipeline.svg)
![Pipeline Architecture](Figures/architecture.svg)
![Iterations of Ensemble Kalman Filter](Figures/EnKF_iterations_Rhone_real.gif)
![Results of Ensemble Kalman Filter compared to glaciological measurements](Figures/mass_loss.svg)
